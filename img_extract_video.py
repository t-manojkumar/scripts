import subprocess
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm

# ---------------- DEVICE SELECTION ----------------

def choose_device():
    print("\nChoose compute device:")

    print("0 → CPU")
    gpu_count = torch.cuda.device_count()

    for i in range(gpu_count):
        print(f"{i+1} → GPU:{i} ({torch.cuda.get_device_name(i)})")

    choice = input("Enter choice number: ").strip()

    if choice == "0":
        return torch.device("cpu")

    try:
        gpu_index = int(choice) - 1
        if 0 <= gpu_index < gpu_count:
            return torch.device(f"cuda:{gpu_index}")
    except:
        pass

    print("[WARN] Invalid choice, falling back to CPU")
    return torch.device("cpu")

device = choose_device()
print(f"[INFO] Using device: {device}")

# ---------------- SCORING ----------------

def score_frame(frame):
    t = torch.from_numpy(frame).to(device).float()

    gray = (
        0.299 * t[..., 0] +
        0.587 * t[..., 1] +
        0.114 * t[..., 2]
    )

    lap = (
        gray[:-2,1:-1] +
        gray[2:,1:-1] +
        gray[1:-1,:-2] +
        gray[1:-1,2:] -
        4 * gray[1:-1,1:-1]
    ).abs()

    sharpness = lap.var()
    contrast = gray.std()
    exposure_penalty = (gray.mean() - 128).abs()

    score = (
        0.6 * sharpness +
        0.3 * contrast -
        0.1 * exposure_penalty
    )

    return score.item()

# ---------------- UTILITIES ----------------

def get_video_resolution(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    w, h = subprocess.check_output(cmd).decode().strip().split(",")
    return int(w), int(h)

def get_video_frame_count(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        return int(subprocess.check_output(cmd).decode().strip())
    except:
        return None

# ---------------- MAIN PIPELINE ----------------

def main():
    video_path = input("\nEnter full video path: ").strip('"')
    out_dir = input("Enter output directory for images: ").strip('"')

    if not os.path.exists(video_path):
        print("[ERROR] Video not found")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("\nChoose extraction mode:")
    print("1 → Extract EVERY frame")
    print("2 → Extract N frames per second")

    mode = input("Enter choice (1 or 2): ").strip()

    fps_filter = ""
    expected_frames = None

    if mode == "2":
        fps = input("Enter frames per second (e.g. 1, 2, 5): ").strip()
        if not fps.isdigit() or int(fps) <= 0:
            print("[ERROR] Invalid FPS")
            return
        fps_filter = f"fps={fps},"
        print(f"[INFO] Extracting {fps} FPS")
    else:
        expected_frames = get_video_frame_count(video_path)
        print("[INFO] Extracting EVERY frame")

    width, height = get_video_resolution(video_path)
    print(f"[INFO] Resolution: {width}x{height}")

    cmd = [
        "ffmpeg",
        "-threads", "0",
        "-i", video_path,
        "-vf", f"{fps_filter}format=rgb24",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-"
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = width * height * 3

    best_score = -1e9
    best_path = None

    progress = tqdm(total=expected_frames, unit="frame")

    idx = 0
    while True:
        raw = pipe.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        filename = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(filename, frame)

        # Downscale for scoring (performance)
        small = cv2.resize(frame, (1280, int(1280 * height / width)))
        score = score_frame(small)

        if score > best_score:
            best_score = score
            best_path = filename

        idx += 1
        progress.update(1)

    progress.close()
    pipe.stdout.close()
    pipe.wait()

    print("\n================ RESULT ================")
    print(f"Total frames extracted : {idx}")
    print(f"Best image file        : {best_path}")
    print(f"Best image score       : {best_score:.2f}")
    print("=======================================\n")

    best_img = cv2.imread(best_path)
    cv2.imshow("BEST IMAGE", best_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    os.startfile(best_path)

# ---------------- ENTRY ----------------

if __name__ == "__main__":
    main()