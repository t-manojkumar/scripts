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


# ── Hardware decoder (probed once) ────────────────────────────────────────────
def detect_hardware_decoder(codec, use_gpu):
    if not use_gpu:
        return None
    candidates = {
        "h264": ["h264_cuvid", "h264_qsv"],
        "hevc": ["hevc_cuvid", "hevc_qsv"],
        "vp9":  ["vp9_cuvid",  "vp9_qsv"],
        "av1":  ["av1_cuvid",  "av1_qsv"],
    }.get(codec, [])
    if not candidates:
        return None
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                           capture_output=True, text=True, timeout=5)
        for dec in candidates:
            if dec in r.stdout:
                return dec
    except Exception:
        pass
    return None


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
    n_gpus  = torch.cuda.device_count()
    print("\nDevice:")
    print("  0 → CPU")
    for gi in range(n_gpus):
        p       = torch.cuda.get_device_properties(gi)
        default = "  [default]" if gi == 0 else ""
        print(f"  {gi + 1} → GPU {gi}: {p.name}  ({p.total_memory // 1024**2} MB VRAM){default}")
    default_dev   = "1" if n_gpus > 0 else "0"
    device_choice = input(f"Choice [{default_dev}]: ").strip() or default_dev

    gpu_idx = int(device_choice) - 1   # 0-based GPU index; -1 means CPU
    use_gpu = (0 <= gpu_idx < n_gpus)
    device  = torch.device(f"cuda:{gpu_idx}" if use_gpu else "cpu")
    if use_gpu:
        p = torch.cuda.get_device_properties(gpu_idx)
        print(f"[INFO] GPU {gpu_idx}: {p.name}  ({p.total_memory // 1024**2} MB VRAM)")
    else:
        print("[INFO] Using CPU")

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

    if top_n:
        heap_mb  = top_n * W * H * 3 / 1024**2
        batch_mb = batch_size * W * H * 3 / 1024**2
        print(f"[INFO] Heap RAM cap : ~{heap_mb:.0f} MB  ({top_n} full frames)")
        print(f"[INFO] Batch RAM    : ~{batch_mb:.0f} MB  ({batch_size} frames)")

    # ── Hardware decoder ──────────────────────────────────────
    hw_dec = detect_hardware_decoder(codec, use_gpu)
    print(f"[INFO] HW decoder   : {hw_dec or 'none (software)'}")

    # ── FFmpeg command ────────────────────────────────────────
    vf_chain = []
    if fps_extract:
        vf_chain.append(f"fps={fps_extract}")
    vf_chain.append("format=bgr24")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-threads", "0"]
    if hw_dec:
        cmd += ["-hwaccel", "cuda", "-c:v", hw_dec]
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

    frame_size = W * H * 3
    raw_buf    = bytearray(frame_size)   # pre-allocated; reused every frame

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
                heap.offer(sc, fi, full_buf[i])
            else:
                fname = os.path.join(out_dir, f"frame_{fi:06d}{ext}")
                if sc > best_score:
                    best_score, best_file = sc, fname
            n_processed += 1
        buf_len = 0
        indices_buf.clear()
        if top_n:
            full_buf.clear()

    # ── Extraction loop ───────────────────────────────────────
    try:
        while True:
            if not read_frame_exact(pipe_process.stdout, raw_buf, frame_size):
                break

            # .copy() detaches from raw_buf before the next readinto overwrites it
            frame_full = np.frombuffer(raw_buf, dtype=np.uint8).reshape(H, W, 3).copy()

            if skip_dupes:
                frame_tiny = cv2.resize(frame_full, (DUPE_W, DUPE_H),
                                        interpolation=cv2.INTER_NEAREST)
                if is_duplicate(frame_tiny, prev_tiny):
                    n_skipped += 1
                    idx += 1
                    pbar.update(1)
                    pbar.set_postfix(skip=n_skipped, keep=n_processed, refresh=False)
                    continue
                prev_tiny = frame_tiny   # old ref freed automatically

            if not top_n:
                fname = os.path.join(out_dir, f"frame_{idx:06d}{ext}")
                pending_writes.append(
                    _write_executor.submit(cv2.imwrite, fname, frame_full, write_params)
                )

            # Resize and copy directly into the pre-allocated (pinned) buffer slot
            cv2.resize(frame_full, (SCORE_W, SCORE_H),
                       dst=score_np[buf_len],          # write directly into slot
                       interpolation=cv2.INTER_AREA)
            indices_buf.append(idx)
            buf_len += 1

            if top_n:
                full_buf.append(frame_full)

            if buf_len == batch_size:
                flush_batch()
                # Trim finished write futures so the list stays bounded
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
