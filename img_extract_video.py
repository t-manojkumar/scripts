import subprocess
import numpy as np
import torch
import cv2
import os
from tqdm import tqdm

# ================= USER OPTIONS =================

def choose_device():
    print("\nChoose compute device:")
    print("0 → CPU")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"{i+1} → GPU:{i} ({torch.cuda.get_device_name(i)})")

    choice = input("Enter choice: ").strip()

    if choice == "0" or not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        idx = int(choice) - 1
        return torch.device(f"cuda:{idx}")
    except:
        print("[WARN] Invalid choice, using CPU")
        return torch.device("cpu")

def choose_extraction_mode():
    print("\nChoose extraction mode:")
    print("1 → Extract EVERY frame")
    print("2 → Extract N frames per second")

    mode = input("Enter choice (1 or 2): ").strip()
    if mode == "1":
        return "all", None

    fps = input("Enter frames per second (e.g. 1, 2, 5): ").strip()
    if not fps.isdigit() or int(fps) <= 0:
        raise ValueError("Invalid FPS value")

    return "fps", int(fps)

# ================= GPU SCORING =================

def score_batch(frames_np, device):
    t = torch.from_numpy(np.stack(frames_np)).to(device).float()
    t = t.permute(0, 3, 1, 2)  # B C H W

    gray = (
        0.299 * t[:, 0] +
        0.587 * t[:, 1] +
        0.114 * t[:, 2]
    )

    # Gradients
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]

    gx_c = gx[:, :-1, :]
    gy_c = gy[:, :, :-1]

    grad_mag = torch.sqrt(gx_c**2 + gy_c**2)

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

# ================= VIDEO UTILS =================

def get_video_info(video):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "csv=p=0",
        video
    ]
    w, h, n = subprocess.check_output(cmd).decode().strip().split(",")
    return int(w), int(h), int(n) if n.isdigit() else None

# ================= MAIN =================

def main():
    device = choose_device()
    print(f"[INFO] Using device: {device}")

    video = input("\nEnter video path: ").strip('"')
    out_dir = input("Enter output directory: ").strip('"')

    os.makedirs(out_dir, exist_ok=True)

    mode, fps = choose_extraction_mode()

    width, height, total = get_video_info(video)
    print(f"[INFO] Resolution: {width}x{height}")

    SCALE_W = 1280
    SCALE_H = int(height * SCALE_W / width)
    BATCH_SIZE = 8 if device.type == "cuda" else 1

    if mode == "fps":
        vf = f"fps={fps},scale={SCALE_W}:-1,format=rgb24"
    else:
        vf = f"scale={SCALE_W}:-1,format=rgb24"

    cmd = [
        "ffmpeg",
        "-c:v", "h264_cuvid",   # GPU decode (safe on Windows)
        "-i", video,
        "-vf", vf,
        "-f", "rawvideo",
        "-"
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = SCALE_W * SCALE_H * 3

    best_score = -1e9
    best_path = None

    batch_frames = []
    batch_paths = []

    pbar = tqdm(unit="frame")
    idx = 0

    while True:
        raw = pipe.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((SCALE_H, SCALE_W, 3))
        path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(path, frame)

        batch_frames.append(frame)
        batch_paths.append(path)

        if len(batch_frames) == BATCH_SIZE:
            scores = score_batch(batch_frames, device)
            for s, p in zip(scores, batch_paths):
                if s > best_score:
                    best_score = s
                    best_path = p
            batch_frames.clear()
            batch_paths.clear()

        idx += 1
        pbar.update(1)

    if batch_frames:
        scores = score_batch(batch_frames, device)
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

# ================= ENTRY =================

if __name__ == "__main__":
    main()
