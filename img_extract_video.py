import subprocess
import numpy as np
import torch
import cv2
import os
import re
import signal
import sys
from tqdm import tqdm
from collections import deque
from pathlib import Path

# GLOBAL STATE FOR CLEANUP
pipe_process = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n[INFO] Interrupted by user. Cleaning up...")
    if pipe_process:
        pipe_process.terminate()
        pipe_process.wait()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# HELPERS
def check_dependencies():
    """Verify required tools are available"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] FFmpeg/FFprobe not found. Please install FFmpeg.")
        sys.exit(1)


def ffprobe(video):
    """Extract video metadata using JSON output for reliability"""
    if not os.path.exists(video):
        raise FileNotFoundError(f"Video file not found: {video}")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,pix_fmt:format=duration",
        "-of", "json",
        video
    ]

    try:
        import json
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode()
        data = json.loads(out)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed: {e.stderr.decode()}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {e}")

    # Extract stream info
    if 'streams' not in data or len(data['streams']) == 0:
        raise RuntimeError("No video stream found in file")

    stream = data['streams'][0]

    codec = stream.get('codec_name', 'unknown')
    w = int(stream.get('width', 0))
    h = int(stream.get('height', 0))
    pix_fmt = stream.get('pix_fmt', 'unknown')

    # Parse frame rate
    fps_str = stream.get('avg_frame_rate', '0/1')
    if '/' in fps_str:
        num, denom = fps_str.split('/')
        fps = float(num) / float(denom) if float(denom) != 0 else 0
    else:
        fps = float(fps_str)

    # Parse duration (try format first, then stream)
    duration = 0
    if 'format' in data and 'duration' in data['format']:
        try:
            duration = float(data['format']['duration'])
        except (ValueError, TypeError):
            pass

    if duration == 0 and 'duration' in stream:
        try:
            duration = float(stream['duration'])
        except (ValueError, TypeError):
            pass

    # If still no duration, estimate from frame count and fps
    if duration == 0 and 'nb_frames' in stream and fps > 0:
        try:
            duration = int(stream['nb_frames']) / fps
        except (ValueError, TypeError):
            pass

    return codec, w, h, fps, duration, pix_fmt


def detect_resume_index(out_dir, ext):
    """Find highest frame index for resume"""
    pattern = re.compile(rf"frame_(\d+){ext}$")
    max_idx = -1
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            m = pattern.match(f)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def detect_hardware_decoder(codec):
    """Detect available hardware decoders"""
    codec_map = {
        'h264': ['h264_cuvid', 'h264_qsv', 'h264_videotoolbox'],
        'hevc': ['hevc_cuvid', 'hevc_qsv', 'hevc_videotoolbox'],
        'vp9': ['vp9_cuvid', 'vp9_qsv'],
        'av1': ['av1_cuvid', 'av1_qsv'],
    }

    decoders = codec_map.get(codec, [])

    for decoder in decoders:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-decoders"],
                capture_output=True, text=True, timeout=2
            )
            if decoder in result.stdout:
                return decoder
        except:
            pass

    return None


# SCORING
def score_batch(frames, device):
    """
    Score frames based on sharpness, contrast, and quality
    Now correctly handles BGR input from OpenCV
    """
    t = torch.from_numpy(np.stack(frames)).to(device).float()
    # Input is BGR from OpenCV resize
    t = t.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Convert BGR to RGB for proper grayscale calculation
    b, g, r = t[:, 0], t[:, 1], t[:, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Gradient magnitude (edge strength)
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]
    gx = gx[:, :-1, :]
    gy = gy[:, :, :-1]
    grad = torch.sqrt(gx ** 2 + gy ** 2)

    # Laplacian variance (sharpness indicator)
    lap = (
            gray[:, :-2, 1:-1] +
            gray[:, 2:, 1:-1] +
            gray[:, 1:-1, :-2] +
            gray[:, 1:-1, 2:] -
            4 * gray[:, 1:-1, 1:-1]
    ).abs()

    # Brightness penalty (prefer well-exposed frames)
    brightness = gray.mean(dim=(1, 2))
    brightness_penalty = (brightness - 128).abs()

    # Detect near-black frames
    black_threshold = 10
    is_black = (brightness < black_threshold).float()

    # Combined score
    score = (
            0.5 * lap.var(dim=(1, 2)) +  # Sharpness
            0.3 * grad.mean(dim=(1, 2)) +  # Edge strength
            0.2 * gray.std(dim=(1, 2)) -  # Contrast
            0.1 * brightness_penalty -  # Exposure
            1000 * is_black  # Heavily penalize black frames
    )

    return score.detach().cpu().tolist()


def is_duplicate_frame(frame, prev_frame, threshold=0.98):
    """Check if frame is nearly identical to previous frame"""
    if prev_frame is None:
        return False

    # Quick pixel difference check
    diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)).mean()
    return diff < (255 * (1 - threshold))


# MAIN
def main():
    global pipe_process

    print("=" * 50)
    print("OPTIMIZED VIDEO FRAME EXTRACTOR")
    print("=" * 50)

    check_dependencies()

    # USER INPUT
    video = input("\nEnter input video path: ").strip('"').strip("'")
    out_dir = input("Enter output folder: ").strip('"').strip("'")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\nChoose output format:")
    print("1 → JPG (smaller, lossy)")
    print("2 → PNG (larger, lossless) - FULL QUALITY")
    print("3 → WEBP (balanced)")
    fmt_choice = input("Choice [default=2]: ").strip() or "2"

    format_map = {
        "1": (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
        "2": (".png", [cv2.IMWRITE_PNG_COMPRESSION, 0]),  # 0 = no compression, max quality
        "3": (".webp", [cv2.IMWRITE_WEBP_QUALITY, 95])
    }
    ext, write_params = format_map.get(fmt_choice, format_map["2"])

    # PROBE VIDEO
    try:
        codec, W, H, fps, duration, pix_fmt = ffprobe(video)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(f"\n{'=' * 50}")
    print(f"Video Info:")
    print(f"  Codec: {codec.upper()}")
    print(f"  Resolution: {W}x{H}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Pixel Format: {pix_fmt}")
    print(f"{'=' * 50}")

    # DEVICE SELECTION
    print("\nChoose processing device:")
    print("0 → CPU")
    print("1 → GPU (if available)")
    device_choice = input("Choice [default=1]: ").strip() or "1"
    device = torch.device("cuda" if device_choice == "1" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using: {device}")

    # EXTRACTION MODE
    print("\nExtraction mode:")
    print("1 → Every frame")
    print("2 → Frames per second (FPS)")
    print("3 → Smart extraction (skip duplicates)")
    mode = input("Choice [default=3]: ").strip() or "3"

    fps_extract = None
    skip_duplicates = False

    if mode == "2":
        fps_extract = float(input("Enter FPS to extract: "))
    elif mode == "3":
        skip_duplicates = True

    # TOP N FRAMES
    print("\nFrame selection:")
    print("1 → Keep all frames")
    print("2 → Keep only top N frames by quality")
    selection = input("Choice [default=2]: ").strip() or "2"

    top_n = None
    if selection == "2":
        top_n = int(input("How many top frames to keep? [default=100]: ").strip() or "100")

    # RESUME
    start_idx = detect_resume_index(out_dir, ext)
    start_time = start_idx / (fps_extract or fps)

    if start_idx > 0:
        print(f"[INFO] Resuming from frame index: {start_idx} (time: {start_time:.2f}s)")

    # SCORING RESOLUTION
    SCORE_W = min(1280, W)
    SCORE_H = int(H * SCORE_W / W)

    batch_size = 32 if device.type == "cuda" else 8

    # HARDWARE ACCELERATION
    hw_decoder = detect_hardware_decoder(codec)
    if hw_decoder and device.type == "cuda":
        print(f"[INFO] Using hardware decoder: {hw_decoder}")

    # BUILD FFMPEG COMMAND
    vf = []
    if fps_extract:
        vf.append(f"fps={fps_extract}")
    vf.append("format=bgr24")  # Use BGR directly for OpenCV

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]

    if hw_decoder and device.type == "cuda":
        cmd += ["-hwaccel", "cuda", "-c:v", hw_decoder]

    if start_time > 0:
        cmd += ["-ss", str(start_time)]

    cmd += ["-i", video]
    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    print(f"[INFO] Starting extraction...")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Scoring resolution: {SCORE_W}x{SCORE_H}")
    if top_n:
        print(f"[INFO] Will keep top {top_n} frames")

    # START FFMPEG PIPE
    pipe_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8
    )

    frame_size = W * H * 3

    # TRACKING
    frame_data = []  # Store (score, idx, fname)
    frames_buf = []  # Scoring buffer
    indices_buf = []

    # For storing full frames when in top_n mode
    full_frames_buffer = {}  # {idx: frame_full}

    prev_frame_small = None

    est_total = int(duration * (fps_extract or fps)) - start_idx if duration > 0 else 1000
    pbar = tqdm(total=max(est_total, 1), desc="Processing", unit="frame", dynamic_ncols=True)

    idx = start_idx
    frames_processed = 0
    frames_skipped = 0

    try:
        while True:
            raw = pipe_process.stdout.read(frame_size)
            if not raw or len(raw) != frame_size:
                break

            # Frame is already in BGR format from FFmpeg
            frame_full = np.frombuffer(raw, np.uint8).reshape((H, W, 3))

            # Downscale for scoring
            frame_small = cv2.resize(frame_full, (SCORE_W, SCORE_H), interpolation=cv2.INTER_AREA)

            # Skip duplicates if enabled
            if skip_duplicates and is_duplicate_frame(frame_small, prev_frame_small):
                frames_skipped += 1
                idx += 1
                pbar.update(1)
                pbar.set_postfix(skipped=frames_skipped, kept=frames_processed)
                continue

            prev_frame_small = frame_small.copy()

            # Add to scoring buffer
            frames_buf.append(frame_small)
            indices_buf.append(idx)

            # Store full frame if in top_n mode
            if top_n:
                full_frames_buffer[idx] = frame_full.copy()

            # Process batch
            if len(frames_buf) == batch_size:
                scores = score_batch(frames_buf, device)

                # Store frame data with scores
                for i, score in enumerate(scores):
                    frame_idx = indices_buf[i]
                    fname = os.path.join(out_dir, f"frame_{frame_idx:06d}{ext}")
                    frame_data.append((score, frame_idx, fname))
                    frames_processed += 1

                frames_buf.clear()
                indices_buf.clear()

            # Write frame immediately if not in top_n mode
            if not top_n:
                fname = os.path.join(out_dir, f"frame_{idx:06d}{ext}")
                cv2.imwrite(fname, frame_full, write_params)

            idx += 1
            pbar.update(1)
            pbar.set_postfix(skipped=frames_skipped, kept=frames_processed)

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        pipe_process.terminate()
        return

    # Process remaining frames
    if frames_buf:
        scores = score_batch(frames_buf, device)
        for i, score in enumerate(scores):
            frame_idx = indices_buf[i]
            fname = os.path.join(out_dir, f"frame_{frame_idx:06d}{ext}")
            frame_data.append((score, frame_idx, fname))
            frames_processed += 1

    pbar.close()
    pipe_process.wait()

    # Initialize best tracking
    best_score = 0
    best_file = "None"

    # WRITE TOP N FRAMES or find best from all
    if top_n and frame_data:
        print(f"\n[INFO] Selecting top {top_n} frames by quality...")

        # Sort by score (descending) and take top N
        frame_data.sort(reverse=True, key=lambda x: x[0])
        top_frames = frame_data[:top_n]

        # Write the top frames from buffered data
        print(f"[INFO] Writing top {len(top_frames)} frames at full resolution...")

        written_count = 0
        for score, frame_idx, fname in tqdm(top_frames, desc="Writing frames", unit="frame"):
            # Check if we have the frame in buffer
            if frame_idx in full_frames_buffer:
                cv2.imwrite(fname, full_frames_buffer[frame_idx], write_params)
                written_count += 1
            else:
                # Fallback: extract from video at timestamp
                timestamp = frame_idx / (fps_extract or fps)
                extract_cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", str(timestamp),
                    "-i", video,
                    "-vframes", "1",
                    "-f", "image2pipe",
                    "-pix_fmt", "bgr24",
                    "-"
                ]

                result = subprocess.run(extract_cmd, capture_output=True)

                if result.returncode == 0 and result.stdout:
                    try:
                        frame_bytes = np.frombuffer(result.stdout, np.uint8).reshape((H, W, 3))
                        cv2.imwrite(fname, frame_bytes, write_params)
                        written_count += 1
                    except:
                        pass

        print(f"[INFO] Successfully wrote {written_count} frames")

        # Clear buffer to free memory
        full_frames_buffer.clear()

        if top_frames:
            best_score = top_frames[0][0]
            best_file = top_frames[0][2]

    elif frame_data:
        # Find best from all processed (when not using top_n mode)
        best_score, best_idx, best_file = max(frame_data, key=lambda x: x[0])

    elif not top_n:
        # No frame_data but frames were written directly
        # Scan output directory to find best
        print("\n[INFO] Scanning written frames to find best...")
        all_written = []
        for f in os.listdir(out_dir):
            if f.endswith(ext):
                fpath = os.path.join(out_dir, f)
                all_written.append(fpath)

        if all_written:
            best_file = all_written[0]
            best_score = -1  # Unknown since not tracked

    # RESULTS
    print(f"\n{'=' * 50}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Frames processed: {frames_processed}")
    if skip_duplicates:
        print(f"Duplicates skipped: {frames_skipped}")
    if top_n:
        print(f"Frames saved: {min(top_n, len(frame_data))}")
    else:
        print(f"Frames saved: {frames_processed}")

    if best_file and best_file != "None":
        print(f"Best frame: {os.path.basename(best_file)}")
        print(f"Best score: {best_score:.2f}")
    else:
        print(f"Best frame: Not determined")

    print(f"Output directory: {out_dir}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()