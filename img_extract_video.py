#!/usr/bin/env python3
"""
GPU-Accelerated Video Frame Extractor
- fp16 CUDA scoring with Sobel/Laplacian conv2d kernels
- Memory-bounded top-N min-heap (O(N) RAM, not O(all frames))
- Async parallel writes via ThreadPoolExecutor
- Pinned memory + non_blocking H2D transfers
- Hardware video decoding (NVDEC/QSV auto-detect)
- Pre-allocated read buffer (no per-frame bytes alloc)
- Pre-warmed CUDA kernels
"""

import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
import re
import signal
import sys
import heapq
import json
import contextlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# ── Global state for cleanup ──────────────────────────────────────────────────
pipe_process = None
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
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"[ERROR] {tool} not found. Please install FFmpeg.")
            sys.exit(1)


# ── FFprobe ───────────────────────────────────────────────────────────────────
def ffprobe(video):
    if not os.path.exists(video):
        raise FileNotFoundError(f"Video not found: {video}")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
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

    s = streams[0]
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
            duration = float(src.get("duration", 0))
            if duration > 0:
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
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=5
        )
        for dec in candidates:
            if dec in result.stdout:
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


# ── Reliable pipe reader (avoids partial-read on large frames) ────────────────
def read_frame_exact(stream, buf, size):
    """Fill buf[:size] from stream. Returns True on success, False on EOF."""
    mv = memoryview(buf)
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
    Keeps the best N (score, frame) pairs in memory.
    Worst-scoring frame is evicted when N is exceeded — O(log N) per insert.
    Peak RAM = N * frame_bytes, not all_frames * frame_bytes.
    """

    def __init__(self, n):
        self.n = n
        self._heap = []    # (score, counter, idx, frame_bgr_or_None)
        self._counter = 0

    def offer(self, score, idx, frame_bgr):
        entry = (score, self._counter, idx, frame_bgr)
        self._counter += 1
        if len(self._heap) < self.n:
            heapq.heappush(self._heap, entry)
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
        # else: below current worst — discard immediately (GC will free frame)

    def sorted_top(self):
        return [(s, idx, fr) for s, _, idx, fr in sorted(self._heap, reverse=True)]

    def __len__(self):
        return len(self._heap)

    def worst_score(self):
        return self._heap[0][0] if self._heap else float("-inf")


# ── Cached convolution kernels (allocated once per device) ───────────────────
_kernels: dict = {}


def _get_kernels(device, dtype):
    key = (device.type, device.index, dtype)
    if key not in _kernels:
        kw = dict(dtype=dtype, device=device)
        _kernels[key] = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], **kw).view(1, 1, 3, 3),  # Sobel-X
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], **kw).view(1, 1, 3, 3),  # Sobel-Y
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], **kw).view(1, 1, 3, 3),    # Laplacian
        )
    return _kernels[key]


# ── GPU batch scoring ─────────────────────────────────────────────────────────
def score_batch(frames_np, device, stream=None):
    """
    Score BGR uint8 frames (list of H×W×3 arrays) on device.
    fp16 on CUDA (~2× memory bandwidth vs fp32), fp32 on CPU.
    Returns list[float] of quality scores.
    """
    arr = np.stack(frames_np)   # (N, H, W, 3) uint8

    ctx = (torch.cuda.stream(stream)
           if (stream is not None and device.type == "cuda")
           else contextlib.nullcontext())

    with ctx:
        if device.type == "cuda":
            # Pinned host memory → fast async DMA to VRAM
            t = torch.from_numpy(arr).pin_memory().to(device, non_blocking=True).half()
            dtype = torch.float16
        else:
            t = torch.from_numpy(arr).float()
            dtype = torch.float32

        # NHWC BGR → NCHW, normalize [0, 1]
        t = t.permute(0, 3, 1, 2).div_(255.0)

        # Perceptual grayscale from BGR channels
        gray = 0.299 * t[:, 2] + 0.587 * t[:, 1] + 0.114 * t[:, 0]  # (N, H, W)
        g    = gray.unsqueeze(1)                                        # (N, 1, H, W)

        sx, sy, lk = _get_kernels(device, dtype)

        lap  = F.conv2d(g, lk, padding=1).squeeze(1).abs()
        gx   = F.conv2d(g, sx, padding=1).squeeze(1)
        gy   = F.conv2d(g, sy, padding=1).squeeze(1)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2))

        brightness = gray.mean(dim=(1, 2))
        is_black   = (brightness < (10.0 / 255.0)).to(dtype)

        score = (
            0.50 * lap.var(dim=(1, 2))          +  # sharpness
            0.30 * grad.mean(dim=(1, 2))         +  # edge strength
            0.20 * gray.std(dim=(1, 2))          -  # contrast
            0.10 * (brightness - 0.5).abs()      -  # exposure
            1000.0 * is_black                       # penalise black frames
        )

        if stream is not None and device.type == "cuda":
            stream.synchronize()

        return score.float().cpu().tolist()


# ── Duplicate detection ───────────────────────────────────────────────────────
def is_duplicate(frame, prev, threshold=0.98):
    if prev is None:
        return False
    diff = np.abs(frame.astype(np.float32) - prev.astype(np.float32)).mean()
    return diff < (255.0 * (1.0 - threshold))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global pipe_process, _write_executor

    print("=" * 56)
    print("  GPU-ACCELERATED VIDEO FRAME EXTRACTOR")
    print("=" * 56)

    check_dependencies()

    # ── User input ────────────────────────────────────────────
    video   = input("\nEnter input video path: ").strip('"').strip("'")
    out_dir = input("Enter output folder:    ").strip('"').strip("'")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\nOutput format:")
    print("  1 → JPG  (smaller, lossy)")
    print("  2 → PNG  (lossless, max quality)  [default]")
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

    print(f"\n{'─' * 56}")
    print(f"  Codec     : {codec.upper()}")
    print(f"  Resolution: {W}×{H}")
    print(f"  FPS       : {fps:.3f}")
    print(f"  Duration  : {duration:.1f}s  ({duration/60:.1f} min)")
    print(f"  Pixel fmt : {pix_fmt}")
    print(f"{'─' * 56}")

    # ── Device selection ──────────────────────────────────────
    cuda_ok = torch.cuda.is_available()
    print(f"\nDevice selection:")
    print("  0 → CPU")
    if cuda_ok:
        print(f"  1 → {torch.cuda.get_device_name(0)}  [default]")
    default_dev = "1" if cuda_ok else "0"
    device_choice = input(f"Choice [{default_dev}]: ").strip() or default_dev
    use_gpu = (device_choice == "1" and cuda_ok)
    device  = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] GPU: {props.name}  VRAM: {props.total_memory // 1024**2} MB")
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

    # ── Tuning ────────────────────────────────────────────────
    SCORE_W    = min(1280, W)
    SCORE_H    = max(1, int(H * SCORE_W / W))
    batch_size = 64 if use_gpu else 16     # larger batches = better GPU utilisation
    n_writers  = min(4, os.cpu_count() or 2)

    if top_n:
        heap_ram_mb   = top_n * W * H * 3 / 1024**2
        batch_ram_mb  = batch_size * W * H * 3 / 1024**2
        print(f"\n[INFO] Heap RAM cap : ~{heap_ram_mb:.0f} MB  ({top_n} full frames)")
        print(f"[INFO] Batch RAM    : ~{batch_ram_mb:.0f} MB  (batch of {batch_size})")

    # ── Hardware decoder ──────────────────────────────────────
    hw_dec = detect_hardware_decoder(codec, use_gpu)
    if hw_dec:
        print(f"[INFO] Hardware decoder : {hw_dec}")
    else:
        print(f"[INFO] Software decoder (no hw accel found for {codec})")

    # ── Build FFmpeg command ───────────────────────────────────
    vf_chain = []
    if fps_extract:
        vf_chain.append(f"fps={fps_extract}")
    vf_chain.append("format=bgr24")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning",
           "-threads", "0"]                 # auto-thread demuxer/decoder

    if hw_dec:
        cmd += ["-hwaccel", "cuda", "-c:v", hw_dec]

    if start_time > 0:
        cmd += ["-ss", str(start_time)]

    cmd += ["-i", video,
            "-vf", ",".join(vf_chain),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    # ── Pre-warm GPU (eliminates first-batch CUDA JIT delay) ──
    if use_gpu:
        _dummy = np.zeros((1, SCORE_H, SCORE_W, 3), dtype=np.uint8)
        _get_kernels(device, torch.float16)
        score_batch([_dummy[0]], device)
        torch.cuda.synchronize()
        print("[INFO] CUDA kernels pre-warmed")

    # ── CUDA stream for pipeline overlap ──────────────────────
    cuda_stream = torch.cuda.Stream() if use_gpu else None

    print(f"\n[INFO] Starting extraction...")
    print(f"[INFO] Batch size   : {batch_size}")
    print(f"[INFO] Score res    : {SCORE_W}×{SCORE_H}")
    print(f"[INFO] Write threads: {n_writers}")
    if top_n:
        print(f"[INFO] Top-N mode   : {top_n} best frames")

    # ── Start FFmpeg pipe ─────────────────────────────────────
    pipe_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    frame_size = W * H * 3
    raw_buf    = bytearray(frame_size)  # pre-allocated; no per-frame alloc

    # ── State ─────────────────────────────────────────────────
    heap        = TopNHeap(top_n) if top_n else None
    frames_buf  = []      # downscaled frames for batch scoring
    indices_buf = []
    full_buf    = []      # full-res frames (top_n mode only)

    best_score  = float("-inf")
    best_file   = None

    prev_small  = None
    idx         = start_idx
    n_processed = 0
    n_skipped   = 0

    _write_executor = ThreadPoolExecutor(max_workers=n_writers)
    pending_writes  = []

    est_total = max(1, int(duration * eff_fps) - start_idx) if duration > 0 else 1000
    pbar = tqdm(total=est_total, desc="Extracting", unit="fr", dynamic_ncols=True)

    # ── Flush scoring batch ───────────────────────────────────
    def flush_batch():
        nonlocal best_score, best_file, n_processed
        if not frames_buf:
            return

        scores = score_batch(frames_buf, device, cuda_stream)

        for i, sc in enumerate(scores):
            fi = indices_buf[i]
            if top_n:
                heap.offer(sc, fi, full_buf[i])
            else:
                # Frame already written; update running best
                fname = os.path.join(out_dir, f"frame_{fi:06d}{ext}")
                if sc > best_score:
                    best_score = sc
                    best_file  = fname
            n_processed += 1

        frames_buf.clear()
        indices_buf.clear()
        if top_n:
            full_buf.clear()

    # ── Extraction loop ───────────────────────────────────────
    try:
        while True:
            if not read_frame_exact(pipe_process.stdout, raw_buf, frame_size):
                break

            # np.frombuffer shares raw_buf; .copy() detaches before next read
            frame_full  = np.frombuffer(raw_buf, dtype=np.uint8).reshape(H, W, 3).copy()
            frame_small = cv2.resize(frame_full, (SCORE_W, SCORE_H), interpolation=cv2.INTER_AREA)

            if skip_dupes and is_duplicate(frame_small, prev_small):
                n_skipped += 1
                idx += 1
                pbar.update(1)
                pbar.set_postfix(skip=n_skipped, keep=n_processed, refresh=False)
                continue

            prev_small = frame_small          # old ref freed; no explicit copy needed

            if not top_n:
                # Write immediately in background; score separately for best-tracking
                fname = os.path.join(out_dir, f"frame_{idx:06d}{ext}")
                pending_writes.append(
                    _write_executor.submit(cv2.imwrite, fname, frame_full, write_params)
                )

            frames_buf.append(frame_small)
            indices_buf.append(idx)
            if top_n:
                full_buf.append(frame_full)

            if len(frames_buf) == batch_size:
                flush_batch()
                # Trim completed futures so list doesn't grow unbounded
                if len(pending_writes) > n_writers * 4:
                    pending_writes[:] = [f for f in pending_writes if not f.done()]

            idx += 1
            pbar.update(1)
            pbar.set_postfix(skip=n_skipped, keep=n_processed, refresh=False)

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        pipe_process.terminate()
        return
    finally:
        pbar.close()

    flush_batch()       # remaining partial batch
    pipe_process.wait()

    # ── Flush async write queue ───────────────────────────────
    if pending_writes:
        print("[INFO] Flushing write queue...")
        failed = 0
        for fut in pending_writes:
            if not fut.result():
                failed += 1
        if failed:
            print(f"[WARN] {failed} frame(s) failed to write")

    _write_executor.shutdown(wait=True)

    # ── Write top-N frames from heap ──────────────────────────
    if top_n and heap:
        top_list = heap.sorted_top()
        print(f"\n[INFO] Writing top {len(top_list)} frames at full resolution...")
        write_futures = []
        with ThreadPoolExecutor(max_workers=n_writers) as pool:
            for sc, fi, frame in top_list:
                fname = os.path.join(out_dir, f"frame_{fi:06d}{ext}")
                write_futures.append((fname, sc, pool.submit(cv2.imwrite, fname, frame, write_params)))
            written = sum(1 for _, _, f in write_futures if f.result())

        print(f"[INFO] Wrote {written} frames")

        if top_list:
            best_score = top_list[0][0]
            best_file  = os.path.join(out_dir, f"frame_{top_list[0][1]:06d}{ext}")

    # ── Summary ───────────────────────────────────────────────
    saved = min(top_n, len(heap)) if (top_n and heap) else n_processed
    print(f"\n{'=' * 56}")
    print("  EXTRACTION COMPLETE")
    print(f"{'─' * 56}")
    print(f"  Frames processed : {n_processed}")
    if skip_dupes:
        print(f"  Duplicates skipped: {n_skipped}")
    print(f"  Frames saved      : {saved}")
    if best_file and best_score > float("-inf"):
        print(f"  Best frame        : {os.path.basename(best_file)}")
        print(f"  Best score        : {best_score:.6f}")
    print(f"  Output directory  : {out_dir}")
    print(f"{'=' * 56}\n")


if __name__ == "__main__":
    main()
