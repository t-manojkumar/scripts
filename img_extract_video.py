import subprocess
import numpy as np
import torch
import cv2
import os
from tqdm import tqdm

# ================= CONFIG =================

BATCH_SIZE = 8                 # Increase to 16 if VRAM allows
DOWNSCALE_WIDTH = 1280         # For scoring only

# =========================================

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

# ---------------- SCORING (BATCHED) ----------------

def score_batch(frames_np):
    t = torch.from_numpy(np.stack(frames_np)).to(device).float()
    t = t.permute(0, 3, 1, 2)  # B C H W

    gray = (
        0.299 * t[:, 0] +
        0.587 * t[:, 1] +
        0.114 * t[:, 2]
    )

    # Gradients
    gx = gray[:, :, 1:] - gray[:, :, :-1]   # B H W-1
    gy = gray[:, 1:, :] - gray[:, :-1, :]   # B H-1 W

    # Align shapes
    gx_c = gx[:, :-1, :]    # B H-1 W-1
    gy_c = gy[:, :, :-1]   # B H-1 W-1

    grad_mag = torch.sqrt(gx_c**2 + gy_c**2)

    # Laplacian (sharpness)
    lap = (
        gray[:, :-2, 1:-1] +
        gray[:, 2:, 1:-1] +
        gray[:, 1:-1, :-2] +
        gray[:, 1:-1, 2:] -
        4 * gray[:, 1:-1, 1:-1]
    ).abs()

    sharpness = lap.var(dim=(1, 2))
    edge_energy = grad_mag.mean(dim=(1, 2))
    contrast = gray.std(dim=(1, 2))
    exposure_penalty = (gray.mean(dim=(1, 2)) - 128).abs()

    score = (
        0.5 * sharpness +
        0.3 * edge_energy +
        0.2 * contrast -
        0.1 * exposure_penalty
    )

    return score.detach().cpu().tolist()

# ---------------- UTILITIES ----------------

def get_video_info(video):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "csv=p=0",
        video
    ]
    out = subprocess.check_output(cmd).decode().strip().split(",")
    w, h = int(out[0]), int(out[1])
    total = int(out[2]) if len(out) > 2 and out[2].isdigit() else None
    return w, h, total

# ---------------- MAIN ----------------

def main():
    video = input("Enter H264 video path: ").strip('"')
    out_dir = input("Enter output directory: ").strip('"')
    fps = input("Frames per second (e.g. 1): ").strip()

    os.makedirs(out_dir, exist_ok=True)

    width, height, total = get_video_info(video)
    print(f"[INFO] {width}x{height}, total frames: {total}")

    cmd = [
        "ffmpeg",
        "-c:v", "h264_cuvid",  # ✅ GPU decode (NVDEC)
        "-i", video,
        "-vf", f"fps={fps},scale={DOWNSCALE_WIDTH}:-1:flags=fast_bilinear,format=rgb24",
        "-f", "rawvideo",
        "-"
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = DOWNSCALE_WIDTH * int(height * DOWNSCALE_WIDTH / width) * 3

    batch_frames = []
    batch_paths = []

    best_score = -1e9
    best_path = None

    idx = 0
    pbar = tqdm(unit="frame")

    while True:
        raw = pipe.stdout.read(frame_size)
        if not raw:
            break

        scaled_h = int(height * DOWNSCALE_WIDTH / width)
        frame = np.frombuffer(raw, np.uint8).reshape((scaled_h, DOWNSCALE_WIDTH, 3))
        filename = os.path.join(out_dir, f"frame_{idx:06d}.jpg")

        cv2.imwrite(filename, frame)
        batch_frames.append(frame)
        batch_paths.append(filename)

        if len(batch_frames) == BATCH_SIZE:
            scores = score_batch(batch_frames)
            for s, p in zip(scores, batch_paths):
                if s > best_score:
                    best_score = s
                    best_path = p
            batch_frames.clear()
            batch_paths.clear()

        idx += 1
        pbar.update(1)

    if batch_frames:
        scores = score_batch(batch_frames)
        for s, p in zip(scores, batch_paths):
            if s > best_score:
                best_score = s
                best_path = p

    pbar.close()
    pipe.wait()

    print("\n================ RESULT ================")
    print(f"Total frames extracted : {idx}")
    print(f"Best image file        : {best_path}")
    print(f"Best image score       : {best_score:.2f}")
    print("=======================================\n")

# ---------------- ENTRY ----------------

if __name__ == "__main__":
    main()
