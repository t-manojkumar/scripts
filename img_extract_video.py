import subprocess
import numpy as np
import torch
import cv2
import os
import psutil
from tqdm import tqdm
import math

# ====================== UTILITIES ======================

def run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()

def analyze_video(video):
    info = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-of", "default=nk=1:nw=1",
        video
    ]).splitlines()

    codec = info[0]
    w, h = int(info[1]), int(info[2])
    fps = eval(info[3])
    frames = int(info[4]) if info[4].isdigit() else None
    duration = float(info[5])

    return codec, w, h, fps, frames, duration

def suggest_device(codec):
    if codec in ("h264", "hevc") and torch.cuda.is_available():
        return "GPU", "H.264/HEVC detected → NVDEC supported"
    return "CPU", f"{codec.upper()} detected → GPU decode not supported"

def gpu_memory_mb():
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return free // (1024 * 1024), total // (1024 * 1024)

# ====================== SCORING ======================

def score_batch(frames, device):
    t = torch.from_numpy(np.stack(frames)).to(device).float()
    t = t.permute(0, 3, 1, 2)

    gray = 0.299*t[:,0] + 0.587*t[:,1] + 0.114*t[:,2]

    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]

    gx = gx[:, :-1, :]
    gy = gy[:, :, :-1]

    grad = torch.sqrt(gx**2 + gy**2)

    lap = (
        gray[:, :-2, 1:-1] +
        gray[:, 2:, 1:-1] +
        gray[:, 1:-1, :-2] +
        gray[:, 1:-1, 2:] -
        4 * gray[:, 1:-1, 1:-1]
    ).abs()

    sharp = lap.var(dim=(1,2))
    edge = grad.mean(dim=(1,2))
    contrast = gray.std(dim=(1,2))
    exposure = (gray.mean(dim=(1,2)) - 128).abs()

    score = 0.5*sharp + 0.3*edge + 0.2*contrast - 0.1*exposure
    return score.detach().cpu().tolist()

# ====================== MAIN ======================

def main():
    video = input("Enter input video path: ").strip('"')
    out_dir = input("Enter output folder: ").strip('"')
    os.makedirs(out_dir, exist_ok=True)

    codec, w, h, fps, total_frames, duration = analyze_video(video)

    print("\n--- Video Analysis ---")
    print(f"Codec     : {codec}")
    print(f"Resolution: {w}x{h}")
    print(f"FPS       : {fps:.2f}")
    print(f"Duration  : {duration:.2f}s")

    suggested, reason = suggest_device(codec)
    print(f"\nSuggested device: {suggested} ({reason})")

    print("\nChoose device:")
    print("0 → CPU")
    if torch.cuda.is_available():
        print("1 → GPU")

    dev_choice = input("Enter choice: ").strip()
    device = torch.device("cuda" if dev_choice == "1" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if device.type == "cuda":
        free, total = gpu_memory_mb()
        print(f"[INFO] GPU Memory: {free}MB free / {total}MB total")

    print("\nChoose extraction mode:")
    print("1 → Every possible frame")
    print("2 → Frames per second")

    mode = input("Enter choice: ").strip()
    fps_extract = None
    if mode == "2":
        fps_extract = int(input("Enter FPS (e.g. 1, 2, 5): "))

    # Scale for scoring
    SCALE_W = 1280
    SCALE_H = int(h * SCALE_W / w)

    batch_size = 8 if device.type == "cuda" else 1

    # Estimate total frames for progress bar
    est_total = int(duration * fps_extract) if fps_extract else total_frames

    # FFmpeg command
    decoder = "h264_cuvid" if codec == "h264" and device.type == "cuda" else None
    cmd = ["ffmpeg"]
    if decoder:
        cmd += ["-c:v", decoder]
    cmd += ["-i", video]

    vf = []
    if fps_extract:
        vf.append(f"fps={fps_extract}")
    vf.append(f"scale={SCALE_W}:-1")
    vf.append("format=rgb24")

    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-"]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = SCALE_W * SCALE_H * 3

    best_score = -1e9
    best_file = None

    frames_buf, paths_buf = [], []

    pbar = tqdm(total=est_total, desc="Extracting frames", unit="frame")

    idx = 0
    while True:
        raw = pipe.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((SCALE_H, SCALE_W, 3))
        fname = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(fname, frame)

        frames_buf.append(frame)
        paths_buf.append(fname)

        if len(frames_buf) == batch_size:
            scores = score_batch(frames_buf, device)
            for s, p in zip(scores, paths_buf):
                if s > best_score:
                    best_score, best_file = s, p
            frames_buf.clear()
            paths_buf.clear()

        idx += 1
        pbar.update(1)

    if frames_buf:
        scores = score_batch(frames_buf, device)
        for s, p in zip(scores, paths_buf):
            if s > best_score:
                best_score, best_file = s, p

    pbar.close()
    pipe.wait()

    print("\n--- RESULT ---")
    print(f"Total frames extracted: {idx}")
    print(f"Best image file       : {best_file}")
    print(f"Best image score      : {best_score:.2f}")

# ====================== ENTRY ======================

if __name__ == "__main__":
    main()