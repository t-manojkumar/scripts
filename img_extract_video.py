import subprocess
import numpy as np
import torch
import cv2
import os
import re
from tqdm import tqdm

#HELPERS
def ffprobe(video):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,duration",
        "-of", "default=nk=1:nw=1",
        video
    ]
    out = subprocess.check_output(cmd).decode().splitlines()
    codec = out[0]
    w, h = int(out[1]), int(out[2])
    fps = eval(out[3])
    duration = float(out[4])
    return codec, w, h, fps, duration

def detect_resume_index(out_dir, ext):
    pattern = re.compile(rf"frame_(\d+){ext}$")
    max_idx = -1
    for f in os.listdir(out_dir):
        m = pattern.match(f)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

#SCORING
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

    score = (
        0.5 * lap.var(dim=(1,2)) +
        0.3 * grad.mean(dim=(1,2)) +
        0.2 * gray.std(dim=(1,2)) -
        0.1 * (gray.mean(dim=(1,2)) - 128).abs()
    )

    return score.detach().cpu().tolist()

#main
def main():
    video = input("Enter input video path: ").strip('"')
    out_dir = input("Enter output folder: ").strip('"')
    os.makedirs(out_dir, exist_ok=True)

    print("\nChoose output format:")
    print("1 → JPG")
    print("2 → PNG")
    print("3 → WEBP")
    fmt_choice = input("Choice: ").strip()
    ext = { "1": ".jpg", "2": ".png", "3": ".webp" }.get(fmt_choice, ".jpg")

    codec, W, H, fps, duration = ffprobe(video)
    print(f"\nVideo: {codec.upper()} | {W}x{H} | {fps:.2f} FPS | {duration:.1f}s")

    print("\nChoose device:")
    print("0 → CPU")
    print("1 → GPU (if available)")
    device = torch.device("cuda" if input("Choice: ") == "1" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using {device}")

    print("\nExtraction mode:")
    print("1 → Every frame")
    print("2 → Frames per second")
    mode = input("Choice: ").strip()
    fps_extract = None
    if mode == "2":
        fps_extract = int(input("Enter FPS: "))

    start_idx = detect_resume_index(out_dir, ext)
    start_time = start_idx / fps if fps_extract is None else start_idx / fps_extract

    print(f"[INFO] Resume from frame index: {start_idx}")

    # Downscale only for scoring
    SCORE_W = 1280
    SCORE_H = int(H * SCORE_W / W)

    batch_size = 8 if device.type == "cuda" else 1

    vf = []
    if fps_extract:
        vf.append(f"fps={fps_extract}")
    vf.append("format=rgb24")

    cmd = ["ffmpeg"]
    if codec == "h264" and device.type == "cuda":
        cmd += ["-c:v", "h264_cuvid"]
    cmd += ["-ss", str(start_time), "-i", video]
    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-"]

    pipe = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    frame_size = W * H * 3
    score_size = SCORE_W * SCORE_H * 3

    best_score = -1e9
    best_file = None

    frames_buf, paths_buf = [], []

    est_total = int(duration * (fps_extract or fps)) - start_idx
    pbar = tqdm(total=max(est_total, 1), desc="Extracting frames", unit="frame", dynamic_ncols=True)

    idx = start_idx
    while True:
        raw = pipe.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((H, W, 3))
        fname = os.path.join(out_dir, f"frame_{idx:06d}{ext}")
        cv2.imwrite(fname, frame)

        # Downscale only for scoring
        small = cv2.resize(frame, (SCORE_W, SCORE_H), interpolation=cv2.INTER_AREA)
        frames_buf.append(small)
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

    print("\n================ RESULT ================")
    print(f"Frames extracted : {idx}")
    print(f"Best image file  : {best_file}")
    print(f"Best image score : {best_score:.2f}")
    print("=======================================\n")

#opening
if __name__ == "__main__":
    main()
