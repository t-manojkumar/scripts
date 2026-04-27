#!/usr/bin/env python3
"""
GPU-Accelerated Video Frame Extractor
Optimizations:
  - Pre-allocated pinned batch tensor → 1 copy per frame (not 2)
  - fp16 CUDA scoring via F.conv2d Sobel/Laplacian kernels
  - torch.no_grad() eliminates autograd graph overhead
  - Fused .to(device, dtype=fp16, non_blocking=True) → single GPU op
  - CUDA stream; .cpu() inside context (no redundant stream.synchronize)
  - Bounded top-N min-heap: O(N) RAM regardless of video length
  - Async parallel writes via ThreadPoolExecutor
  - 128px thumbnail for duplicate detection (100x fewer pixels than score res)
  - Hardware video decoding (NVDEC/QSV, probed once)
  - Pre-warmed CUDA kernels (no first-batch JIT stall)
  - Reliable readinto loop (no partial-frame reads)
  - FFmpeg -threads 0 (auto-threaded demux/decode)
"""

import contextlib
import heapq
import json
import os
import re
import signal
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Global state for signal-handler cleanup ───────────────────────────────────
pipe_process    = None
_write_executor = None


def signal_handler(sig, frame):
    print("\n\n[INFO] Interrupted. Cleaning up...")
    if pipe_process:
        pipe_process.terminate()
        pipe_process.wait()
    if _write_executor:
        _write_executor.shutdown(wait=False)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# ── Dependency check ──────────────────────────────────────────────────────────
def check_dependencies():
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"[ERROR] {tool} not found. Please install FFmpeg.")
            sys.exit(1)


# ── FFprobe ───────────────────────────────────────────────────────────────────
def ffprobe(video):
    if not os.path.exists(video):
        raise FileNotFoundError(f"Video not found: {video}")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,pix_fmt,nb_frames"
        ":format=duration",
        "-of", "json", video,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode()
        data = json.loads(out)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed: {e.stderr.decode()}")

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found")
    s       = streams[0]
    codec   = s.get("codec_name", "unknown")
    W, H    = int(s.get("width", 0)), int(s.get("height", 0))
    pix_fmt = s.get("pix_fmt", "unknown")

    fps_str = s.get("avg_frame_rate", "0/1")
    if "/" in fps_str:
        n, d = fps_str.split("/")
        fps = float(n) / float(d) if float(d) else 0.0
    else:
        fps = float(fps_str)

    duration = 0.0
    for src in (data.get("format", {}), s):
        try:
            v = float(src.get("duration", 0))
            if v > 0:
                duration = v
                break
        except (ValueError, TypeError):
            pass

    if duration == 0 and "nb_frames" in s and fps > 0:
        try:
            duration = int(s["nb_frames"]) / fps
        except (ValueError, TypeError):
            pass

    return codec, W, H, fps, duration, pix_fmt


# ── GPU enumeration (torch.cuda primary, nvidia-smi fallback) ────────────────
def enumerate_gpus():
    """
    Return a list of dicts describing available GPUs.
    Tries torch.cuda first; falls back to nvidia-smi so eGPUs and GPUs with
    mismatched CUDA drivers still appear in the menu with a useful warning.
    Each entry also carries compute_cap so NVDEC capability checks work even
    when PyTorch is CPU-only.
    """
    gpus = []

    # Primary: torch.cuda (guaranteed to work for both NVDEC and scoring)
    n = torch.cuda.device_count()
    for i in range(n):
        p   = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        gpus.append(dict(
            label       = f"GPU {i}: {p.name}  ({p.total_memory // 1024**2} MB VRAM)",
            name        = p.name,
            cuda_idx    = i,
            cuda_ok     = True,
            compute_cap = cap,
        ))

    # Fallback: nvidia-smi — catches GPUs visible to the driver but not to PyTorch.
    # NVDEC / scale_cuda / hwupload_cuda still work via FFmpeg in this mode.
    if not gpus:
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 3:
                        continue
                    vram = int(parts[2]) if parts[2].isdigit() else 0
                    cap  = None
                    if len(parts) >= 4 and "." in parts[3]:
                        try:
                            cap = tuple(int(x) for x in parts[3].split("."))
                        except ValueError:
                            cap = None
                    if cap is None:
                        cap = _infer_compute_cap_from_name(parts[1])
                    gpus.append(dict(
                        label       = f"GPU {parts[0]}: {parts[1]}  ({vram} MB VRAM)"
                                       "  ⚠ PyTorch CUDA missing — FFmpeg NVDEC still usable",
                        name        = parts[1],
                        cuda_idx    = None,
                        cuda_ok     = False,
                        compute_cap = cap,
                    ))
        except Exception:
            pass

    return gpus


def _infer_compute_cap_from_name(name):
    """Best-effort compute-capability guess from GPU model name (driver < 515)."""
    n = name.lower()
    if "rtx 50" in n or "rtx 40" in n or "ada" in n or "h100" in n or "h200" in n:
        return (8, 9)
    if "rtx 30" in n or "a100" in n or "a40" in n or "a10" in n or "ampere" in n:
        return (8, 6)
    if "rtx 20" in n or "gtx 16" in n or "titan rtx" in n or "turing" in n:
        return (7, 5)
    if "v100" in n or "titan v" in n or "volta" in n:
        return (7, 0)
    if "gtx 10" in n or "titan x" in n or "p100" in n or "p40" in n or "pascal" in n:
        return (6, 1)
    if "gtx 9" in n or "maxwell" in n or "m40" in n or "m60" in n:
        return (5, 2)
    return (5, 0)   # conservative default — base NVDEC (H.264) only


# ── Hardware decoder — capability-aware, PyTorch-independent ─────────────────
# NVIDIA NVDEC capability matrix:
#   5.0  Maxwell-1: H.264
#   5.2  Maxwell-2: + HEVC 8-bit
#   6.0+ Pascal   : + HEVC 10-bit, VP9
#   7.0+ Volta    : (refinements)
#   7.5  Turing   : (1660 Ti, RTX 20xx)  H.264, HEVC, VP9   — NO AV1
#   8.6+ Ampere   : + AV1
#   8.9  Ada      : + AV1 (faster)
def detect_hardware_decoder(codec, gpu_info):
    """
    gpu_info : dict from enumerate_gpus()  OR  None (no NVIDIA GPU).
    Works whether or not PyTorch CUDA is available — only requires the driver.
    Returns (decoder_name | None, skip_reason | None).
    """
    if not gpu_info:
        return None, None

    cap   = gpu_info.get("compute_cap") or (5, 0)
    name  = gpu_info.get("name", "GPU")

    # Per-codec NVDEC capability check
    min_cap = {
        "h264": (5, 0),
        "hevc": (5, 2),
        "vp9":  (6, 0),
        "av1":  (8, 6),
    }.get(codec)

    if min_cap is None:
        return None, None     # codec not in NVDEC table

    if cap < min_cap:
        gen_name = {(8, 6): "Ampere (RTX 30xx)",
                    (6, 0): "Pascal (GTX 10xx)",
                    (5, 2): "Maxwell 2 (GTX 9xx)"}.get(min_cap, f"compute {min_cap[0]}.{min_cap[1]}")
        return None, (f"{name} (compute {cap[0]}.{cap[1]}) cannot {codec.upper()} NVDEC. "
                      f"Needs {gen_name} or newer.")

    # Verify FFmpeg has the decoder compiled
    candidates = {
        "h264": ["h264_cuvid", "h264_qsv"],
        "hevc": ["hevc_cuvid", "hevc_qsv"],
        "vp9":  ["vp9_cuvid",  "vp9_qsv"],
        "av1":  ["av1_cuvid",  "av1_qsv"],
    }.get(codec, [])

    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                           capture_output=True, text=True, timeout=5)
        for dec in candidates:
            if dec in r.stdout:
                return dec, None
    except Exception:
        pass
    return None, None


def has_ffmpeg_filter(name):
    """Check whether FFmpeg has a given filter compiled in (e.g. 'scale_cuda')."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                           capture_output=True, text=True, timeout=5)
        return name in r.stdout
    except Exception:
        return False


def has_libdav1d():
    """libdav1d is a much faster AV1 software decoder than libaom-av1."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                           capture_output=True, text=True, timeout=5)
        return "libdav1d" in r.stdout
    except Exception:
        return False


# ── Resume detection ──────────────────────────────────────────────────────────
def detect_resume_index(out_dir, ext):
    pattern = re.compile(rf"frame_(\d+){re.escape(ext)}$")
    max_idx = -1
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            m = pattern.match(f)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


# ── Reliable pipe reader ──────────────────────────────────────────────────────
def read_frame_exact(stream, buf, size):
    """Fill buf[:size] from pipe. Returns True on success, False on EOF."""
    mv  = memoryview(buf)
    pos = 0
    while pos < size:
        n = stream.readinto(mv[pos:size])
        if not n:
            return False
        pos += n
    return True


# ── Bounded top-N min-heap ────────────────────────────────────────────────────
class TopNHeap:
    """
    Retains only the best N frames by score.
    Evicts the worst entry when a higher-scored frame arrives.
    Peak RAM = N × frame_bytes  (independent of video length).
    """

    def __init__(self, n):
        self.n        = n
        self._heap    = []
        self._counter = 0

    def offer(self, score, idx, frame_bgr):
        entry = (score, self._counter, idx, frame_bgr)
        self._counter += 1
        if len(self._heap) < self.n:
            heapq.heappush(self._heap, entry)
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
        # else: below current worst → discarded, frame GC'd immediately

    def sorted_top(self):
        return [(s, idx, fr) for s, _, idx, fr in sorted(self._heap, reverse=True)]

    def __len__(self):
        return len(self._heap)


# ── Kernel cache (allocated once per device/dtype) ────────────────────────────
_kernels: dict = {}


def _get_kernels(device, dtype):
    key = (str(device), dtype)
    if key not in _kernels:
        kw = dict(dtype=dtype, device=device)
        _kernels[key] = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], **kw).view(1, 1, 3, 3),
            torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], **kw).view(1, 1, 3, 3),
            torch.tensor([[ 0, 1, 0], [ 1,-4, 1], [ 0, 1, 0]], **kw).view(1, 1, 3, 3),
        )
    return _kernels[key]   # Sobel-X, Sobel-Y, Laplacian


# ── Batch scoring ─────────────────────────────────────────────────────────────
def score_batch(buf, n, device, stream=None):
    """
    buf    : (B, H, W, 3) uint8 — pinned torch.Tensor (CUDA) or np.ndarray (CPU).
    n      : number of valid frames in buf.
    stream : optional torch.cuda.Stream for pipeline overlap.

    fp16 on CUDA (~2× memory bandwidth), fp32 on CPU.
    torch.no_grad() prevents autograd graph construction.
    Fused .to(device, dtype=fp16) is a single GPU kernel (not two).
    .cpu() inside the stream context is the correct sync point — no explicit
    stream.synchronize() needed.
    """
    ctx = (torch.cuda.stream(stream)
           if (stream is not None and device.type == "cuda")
           else contextlib.nullcontext())

    with torch.no_grad(), ctx:
        if device.type == "cuda":
            # buf is pre-allocated pinned memory → non_blocking DMA + fused fp16 cast
            t = buf[:n].to(device=device, dtype=torch.float16, non_blocking=True)
        else:
            t = torch.from_numpy(np.asarray(buf[:n])).float()

        # NHWC BGR → NCHW, normalize [0, 1]
        t    = t.permute(0, 3, 1, 2).div_(255.0)
        gray = 0.299 * t[:, 2] + 0.587 * t[:, 1] + 0.114 * t[:, 0]
        g    = gray.unsqueeze(1)

        sx, sy, lk = _get_kernels(device, t.dtype)

        lap  = F.conv2d(g, lk, padding=1).squeeze(1).abs()
        gx   = F.conv2d(g, sx, padding=1).squeeze(1)
        gy   = F.conv2d(g, sy, padding=1).squeeze(1)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2))

        brightness = gray.mean(dim=(1, 2))
        is_black   = (brightness < (10.0 / 255.0)).to(t.dtype)

        score = (
            0.50 * lap.var(dim=(1, 2))      +   # sharpness
            0.30 * grad.mean(dim=(1, 2))    +   # edge strength
            0.20 * gray.std(dim=(1, 2))     -   # contrast
            0.10 * (brightness - 0.5).abs() -   # exposure penalty
            1000.0 * is_black                   # black-frame penalty
        )

        # .cpu() inside the stream context → correctly ordered D2H DMA (no separate sync)
        return score.float().cpu().tolist()


# ── Duplicate detection ───────────────────────────────────────────────────────
def is_duplicate(frame, prev, threshold=0.98):
    if prev is None:
        return False
    return (np.abs(frame.astype(np.float32) - prev.astype(np.float32)).mean()
            < 255.0 * (1.0 - threshold))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global pipe_process, _write_executor

    print("=" * 58)
    print("  GPU-ACCELERATED VIDEO FRAME EXTRACTOR")
    print("=" * 58)

    check_dependencies()

    # ── User input ────────────────────────────────────────────
    video   = input("\nEnter input video path: ").strip('"').strip("'")
    out_dir = input("Enter output folder:    ").strip('"').strip("'")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\nOutput format:")
    print("  1 → JPG  (smaller, lossy)")
    print("  2 → PNG  (lossless)  [default]")
    print("  3 → WEBP (balanced)")
    fmt_choice = input("Choice [2]: ").strip() or "2"
    format_map = {
        "1": (".jpg",  [cv2.IMWRITE_JPEG_QUALITY,    95]),
        "2": (".png",  [cv2.IMWRITE_PNG_COMPRESSION,  0]),
        "3": (".webp", [cv2.IMWRITE_WEBP_QUALITY,    95]),
    }
    ext, write_params = format_map.get(fmt_choice, format_map["2"])

    # ── Probe ─────────────────────────────────────────────────
    try:
        codec, W, H, fps, duration, pix_fmt = ffprobe(video)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(f"\n{'─' * 58}")
    print(f"  {codec.upper()}  {W}×{H}  {fps:.3f} fps  {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Pixel format: {pix_fmt}")
    print(f"{'─' * 58}")

    # ── Device ────────────────────────────────────────────────
    gpus = enumerate_gpus()
    print("\nDevice:")
    print("  0 → CPU")
    for i, g in enumerate(gpus):
        tag = "  [default]" if i == 0 and g["cuda_ok"] else ""
        print(f"  {i + 1} → {g['label']}{tag}")

    has_cuda = any(g["cuda_ok"] for g in gpus)
    default_dev   = "1" if has_cuda else "0"
    device_choice = input(f"Choice [{default_dev}]: ").strip() or default_dev

    try:
        choice_i = int(device_choice)
    except ValueError:
        choice_i = 0

    # nvidia_gpu = the selected NVIDIA GPU dict (used for FFmpeg NVDEC/scale_cuda
    #              even when PyTorch CUDA is unavailable).
    # use_gpu    = True only when PyTorch can actually run scoring on CUDA.
    nvidia_gpu = None
    if choice_i == 0 or not gpus or choice_i > len(gpus):
        use_gpu = False
        gpu_idx = -1
        device  = torch.device("cpu")
        print("[INFO] PyTorch scoring : CPU")
    else:
        g          = gpus[choice_i - 1]
        nvidia_gpu = g           # FFmpeg can still use this GPU regardless of PyTorch
        if g["cuda_ok"]:
            gpu_idx = g["cuda_idx"]
            use_gpu = True
            device  = torch.device(f"cuda:{gpu_idx}")
            p       = torch.cuda.get_device_properties(gpu_idx)
            print(f"[INFO] PyTorch scoring : GPU {gpu_idx}  ({p.name}, "
                  f"{p.total_memory // 1024**2} MB VRAM)")
        else:
            # PyTorch is CPU-only, but NVDEC + scale_cuda will still run on the GPU.
            print(f"\n[WARN] PyTorch can't see CUDA on this install (running on CPU).")
            print(f"[WARN]   Installed torch version : {torch.__version__}")
            print(f"[WARN]   To enable GPU scoring, in this Python:")
            print(f"[WARN]     pip uninstall torch torchvision -y")
            print(f"[WARN]     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print(f"[INFO] Continuing — FFmpeg will still use {g['name']} for "
                  f"NVDEC + GPU scaling where possible.")
            use_gpu = False
            gpu_idx = -1
            device  = torch.device("cpu")

    # ── Extraction mode ───────────────────────────────────────
    print("\nExtraction mode:")
    print("  1 → Every frame")
    print("  2 → Fixed FPS")
    print("  3 → Smart (skip near-duplicates)  [default]")
    mode = input("Choice [3]: ").strip() or "3"
    fps_extract = None
    skip_dupes  = False
    if mode == "2":
        fps_extract = float(input("Extract FPS: "))
    elif mode == "3":
        skip_dupes = True

    # ── Frame selection ───────────────────────────────────────
    print("\nFrame selection:")
    print("  1 → Keep all frames")
    print("  2 → Keep top N by quality  [default]")
    selection = input("Choice [2]: ").strip() or "2"
    top_n = None
    if selection == "2":
        top_n = int(input("Top N frames [100]: ").strip() or "100")

    # ── Resume ────────────────────────────────────────────────
    start_idx  = detect_resume_index(out_dir, ext)
    eff_fps    = fps_extract or fps
    start_time = start_idx / eff_fps if eff_fps > 0 else 0.0
    if start_idx > 0:
        print(f"[INFO] Resuming from frame {start_idx} (t={start_time:.2f}s)")

    # ── Sizes & tuning ────────────────────────────────────────
    SCORE_W    = min(1280, W)
    SCORE_H    = max(1, int(H * SCORE_W / W))
    DUPE_W     = min(128, W)                   # 128px thumbnail for duplicate check
    DUPE_H     = max(1, int(H * DUPE_W / W))  # ~100× fewer pixels than score res
    batch_size = 64 if use_gpu else 16
    n_writers  = min(4, os.cpu_count() or 2)

    # ── Two-pass strategy decision ────────────────────────────
    # When top_n is set AND source resolution > score resolution, ask FFmpeg
    # to pre-scale on its side (multi-threaded). Pipe data drops e.g. 9× for
    # 4K→720p, eliminating the per-frame Python cv2.resize on full frames and
    # the multi-GB full-res heap. Top-N frames are re-extracted at full res
    # at the end via a single FFmpeg select-filter pass.
    use_two_pass = bool(top_n) and (W > SCORE_W or H > SCORE_H)
    PIPE_W, PIPE_H = (SCORE_W, SCORE_H) if use_two_pass else (W, H)

    if top_n:
        if use_two_pass:
            heap_mb  = top_n * 0.001     # heap stores (score, idx) only — no frame data
            batch_mb = batch_size * SCORE_W * SCORE_H * 3 / 1024**2
            print(f"[INFO] Two-pass mode: pipe @ {PIPE_W}×{PIPE_H} (FFmpeg-side scale)")
            print(f"[INFO] Pipe data    : ~{(W*H*3)/(SCORE_W*SCORE_H*3):.0f}× smaller "
                  f"({W*H*3/1024**2:.0f} → {SCORE_W*SCORE_H*3/1024**2:.1f} MB/frame)")
        else:
            heap_mb  = top_n * W * H * 3 / 1024**2
            batch_mb = batch_size * W * H * 3 / 1024**2
        print(f"[INFO] Heap RAM cap : ~{heap_mb:.0f} MB")
        print(f"[INFO] Batch RAM    : ~{batch_mb:.0f} MB  ({batch_size} frames)")

    # ── Hardware decoder (PyTorch-independent) ────────────────
    # NVDEC works whenever the driver and codec/capability allow it,
    # regardless of whether PyTorch can use CUDA.
    hw_dec, hw_skip = detect_hardware_decoder(codec, nvidia_gpu)
    if hw_skip:
        print(f"[WARN] {hw_skip}")

    # GPU-side scaling: scale_cuda lets FFmpeg keep frames in VRAM end-to-end
    # (NVDEC → scale_cuda → hwdownload → bgr24). Avoids both CPU scaling and
    # the per-frame Python cv2.resize.
    use_scale_cuda = bool(hw_dec) and has_ffmpeg_filter("scale_cuda")

    # Software-decoder optimization: prefer libdav1d for AV1 (2–3× libaom-av1)
    sw_codec_override = None
    if not hw_dec and codec == "av1" and has_libdav1d():
        sw_codec_override = "libdav1d"

    # Status banner
    if hw_dec:
        chain = f"NVDEC ({hw_dec})"
        if use_scale_cuda and (PIPE_W, PIPE_H) != (W, H):
            chain += " → scale_cuda → hwdownload"
        print(f"[INFO] Decode chain : {chain}  (GPU)")
    else:
        decoder_label = sw_codec_override or "default"
        print(f"[INFO] Decode chain : {decoder_label}  (CPU — NVDEC unavailable for this codec/GPU)")

    # ── FFmpeg command ────────────────────────────────────────
    vf_chain = []
    if fps_extract:
        vf_chain.append(f"fps={fps_extract}")

    if use_scale_cuda:
        # NVDEC outputs CUDA frames; scale on GPU, then hwdownload to CPU.
        if (PIPE_W, PIPE_H) != (W, H):
            vf_chain.append(f"scale_cuda={PIPE_W}:{PIPE_H}")
        vf_chain.append("hwdownload")
        vf_chain.append("format=bgr24")
    else:
        # CPU scale (still multi-threaded inside FFmpeg via libswscale).
        if (PIPE_W, PIPE_H) != (W, H):
            vf_chain.append(f"scale={PIPE_W}:{PIPE_H}:flags=area")
        vf_chain.append("format=bgr24")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-threads", "0"]
    if hw_dec:
        cmd += ["-hwaccel", "cuda"]
        if use_scale_cuda:
            cmd += ["-hwaccel_output_format", "cuda"]   # keep NVDEC frames in VRAM
        cmd += ["-c:v", hw_dec]
    elif sw_codec_override:
        cmd += ["-c:v", sw_codec_override]

    if start_time > 0:
        cmd += ["-ss", str(start_time)]

    cmd += ["-i", video, "-vf", ",".join(vf_chain),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    # ── Pre-allocated batch buffer ────────────────────────────
    # GPU: pinned torch tensor → single copy per frame + DMA to VRAM
    # CPU: plain ndarray      → zero-copy torch.from_numpy in score_batch
    if use_gpu:
        torch.cuda.set_device(gpu_idx)   # bind all CUDA ops to the selected GPU
        score_pin = torch.zeros((batch_size, SCORE_H, SCORE_W, 3),
                                 dtype=torch.uint8).pin_memory()
        score_np  = score_pin.numpy()   # CPU-side view for cv2 writes
        # Pre-warm CUDA kernels with the actual buffer (eliminates first-batch JIT)
        _get_kernels(device, torch.float16)
        score_batch(score_pin, 1, device)
        torch.cuda.synchronize(device)
        print("[INFO] CUDA kernels pre-warmed")
        cuda_stream = torch.cuda.Stream(device=device)
        buf_arg     = score_pin
    else:
        score_pin   = None
        score_np    = np.empty((batch_size, SCORE_H, SCORE_W, 3), dtype=np.uint8)
        cuda_stream = None
        buf_arg     = score_np

    print(f"\n[INFO] Starting extraction...")
    print(f"[INFO] Batch size   : {batch_size}  |  Score res  : {SCORE_W}×{SCORE_H}")
    print(f"[INFO] Write threads: {n_writers}   |  Dupe res   : {DUPE_W}×{DUPE_H}")
    if top_n:
        print(f"[INFO] Top-N heap   : {top_n} frames")

    # ── Start FFmpeg pipe ─────────────────────────────────────
    pipe_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    frame_size = PIPE_W * PIPE_H * 3        # bytes per piped frame (post-scale)
    raw_buf    = bytearray(frame_size)      # pre-allocated; reused every frame
    pipe_at_score_res = (PIPE_W == SCORE_W and PIPE_H == SCORE_H)

    # ── State ─────────────────────────────────────────────────
    heap        = TopNHeap(top_n) if top_n else None
    indices_buf = []
    full_buf    = []     # full-res frames kept for top_n heap only
    buf_len     = 0      # fill level of score_np / score_pin

    best_score  = float("-inf")
    best_file   = None
    prev_tiny   = None
    idx         = start_idx
    n_processed = 0
    n_skipped   = 0

    _write_executor = ThreadPoolExecutor(max_workers=n_writers)
    pending_writes  = []

    est_total = max(1, int(duration * eff_fps) - start_idx) if duration > 0 else 1000
    pbar = tqdm(total=est_total, desc="Extracting", unit="fr", dynamic_ncols=True)

    # ── Flush scoring batch ───────────────────────────────────
    def flush_batch():
        nonlocal buf_len, best_score, best_file, n_processed
        if buf_len == 0:
            return
        scores = score_batch(buf_arg, buf_len, device, cuda_stream)
        for i, sc in enumerate(scores):
            fi = indices_buf[i]
            if top_n:
                # Two-pass: store metadata only (None) — frame is re-extracted later.
                # One-pass: store the full-res frame from full_buf.
                heap.offer(sc, fi, None if use_two_pass else full_buf[i])
            else:
                fname = os.path.join(out_dir, f"frame_{fi:06d}{ext}")
                if sc > best_score:
                    best_score, best_file = sc, fname
            n_processed += 1
        buf_len = 0
        indices_buf.clear()
        if top_n and not use_two_pass:
            full_buf.clear()

    # ── Extraction loop ───────────────────────────────────────
    try:
        while True:
            if not read_frame_exact(pipe_process.stdout, raw_buf, frame_size):
                break

            # frame_pipe is at PIPE resolution (= SCORE res in two-pass mode,
            # = source res otherwise). .copy() detaches from raw_buf before the
            # next readinto overwrites it.
            frame_pipe = np.frombuffer(raw_buf, dtype=np.uint8).reshape(PIPE_H, PIPE_W, 3).copy()

            if skip_dupes:
                # Reuse frame_pipe directly if already small enough
                if PIPE_W <= DUPE_W * 2:
                    frame_tiny = frame_pipe
                else:
                    frame_tiny = cv2.resize(frame_pipe, (DUPE_W, DUPE_H),
                                            interpolation=cv2.INTER_NEAREST)
                if is_duplicate(frame_tiny, prev_tiny):
                    n_skipped += 1
                    idx += 1
                    pbar.update(1)
                    pbar.set_postfix(skip=n_skipped, keep=n_processed, refresh=False)
                    continue
                prev_tiny = frame_tiny

            if not top_n:
                # All-frames mode: pipe is at full res, save as-is
                fname = os.path.join(out_dir, f"frame_{idx:06d}{ext}")
                pending_writes.append(
                    _write_executor.submit(cv2.imwrite, fname, frame_pipe, write_params)
                )

            # Fill the pre-allocated (pinned) scoring buffer slot
            if pipe_at_score_res:
                # Already at score resolution — direct memcpy into pinned slot
                np.copyto(score_np[buf_len], frame_pipe)
            else:
                # Resize on CPU into the pinned slot (small overhead vs 4K cv2.resize)
                cv2.resize(frame_pipe, (SCORE_W, SCORE_H),
                           dst=score_np[buf_len],
                           interpolation=cv2.INTER_AREA)
            indices_buf.append(idx)
            buf_len += 1

            # Only buffer full-res frames when we have them AND need them for
            # the heap (i.e., source already ≤ score res, no two-pass needed)
            if top_n and not use_two_pass:
                full_buf.append(frame_pipe)

            if buf_len == batch_size:
                flush_batch()
                if len(pending_writes) > n_writers * 4:
                    pending_writes[:] = [f for f in pending_writes if not f.done()]

            idx += 1
            pbar.update(1)
            pbar.set_postfix(skip=n_skipped, keep=n_processed, refresh=False)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        pipe_process.terminate()
        return
    finally:
        pbar.close()

    flush_batch()
    pipe_process.wait()

    # ── Flush async write queue ───────────────────────────────
    if pending_writes:
        print("[INFO] Flushing write queue...")
        failed = sum(1 for f in pending_writes if not f.result())
        if failed:
            print(f"[WARN] {failed} frame(s) failed to write")
    _write_executor.shutdown(wait=True)

    # ── Write top-N frames ────────────────────────────────────
    if top_n and heap:
        top_list = heap.sorted_top()

        if use_two_pass:
            # Single-pass select-filter extraction at full source resolution.
            # FFmpeg decodes once and emits only the requested frames, in order.
            indices_sorted = sorted({fi for _, fi, _ in top_list})
            print(f"\n[INFO] Pass 2: extracting top {len(indices_sorted)} frames "
                  f"at {W}×{H} via FFmpeg select filter...")

            # Build select expression: eq(n,X)+eq(n,Y)+...  (commas escaped for filter syntax)
            select_expr = "+".join(f"eq(n\\,{i})" for i in indices_sorted)

            tmp_pattern = os.path.join(out_dir, f"_tmp_%06d{ext}")
            extract_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "0",
            ]
            if hw_dec:
                extract_cmd += ["-hwaccel", "cuda", "-c:v", hw_dec]
            elif sw_codec_override:
                extract_cmd += ["-c:v", sw_codec_override]
            extract_cmd += [
                "-i", video,
                "-vf", f"select='{select_expr}'",
                "-vsync", "vfr",
                "-start_number", "0",
                "-y", tmp_pattern,
            ]
            r = subprocess.run(extract_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[ERROR] Pass-2 extraction failed: {r.stderr}")
                written = 0
            else:
                # FFmpeg emits frames in source-order matching indices_sorted
                written = 0
                for i, fi in enumerate(indices_sorted):
                    src = os.path.join(out_dir, f"_tmp_{i:06d}{ext}")
                    dst = os.path.join(out_dir, f"frame_{fi:06d}{ext}")
                    if os.path.exists(src):
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.rename(src, dst)
                        written += 1
        else:
            print(f"\n[INFO] Writing top {len(top_list)} frames at full resolution...")
            with ThreadPoolExecutor(max_workers=n_writers) as pool:
                futs = [
                    pool.submit(cv2.imwrite,
                                os.path.join(out_dir, f"frame_{fi:06d}{ext}"),
                                fr, write_params)
                    for _, fi, fr in top_list
                ]
                written = sum(1 for f in futs if f.result())

        print(f"[INFO] Wrote {written} frames")
        if top_list:
            best_score = top_list[0][0]
            best_file  = os.path.join(out_dir, f"frame_{top_list[0][1]:06d}{ext}")

    # ── Summary ───────────────────────────────────────────────
    saved = min(top_n, len(heap)) if (top_n and heap) else n_processed
    print(f"\n{'=' * 58}")
    print("  EXTRACTION COMPLETE")
    print(f"{'─' * 58}")
    print(f"  Frames processed  : {n_processed}")
    if skip_dupes:
        print(f"  Duplicates skipped: {n_skipped}")
    print(f"  Frames saved      : {saved}")
    if best_file and best_score > float("-inf"):
        print(f"  Best frame        : {os.path.basename(best_file)}")
        print(f"  Best score        : {best_score:.6f}")
    print(f"  Output directory  : {out_dir}")
    print(f"{'=' * 58}\n")


if __name__ == "__main__":
    main()
