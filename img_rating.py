import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

BLUR_THRESHOLD = 100.0      # lower = more tolerant
MAX_WORKERS = os.cpu_count()

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def rate_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Blur detection (early skip) ----
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < BLUR_THRESHOLD:
        return None  # skip blurry image

    # ---- Face detection ----
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_score = 1 if len(faces) > 0 else 0

    # ---- Quality metrics ----
    contrast = gray.std()

    brightness = gray.mean()
    brightness_score = 255 - abs(brightness - 127)

    noise = np.std(gray - cv2.GaussianBlur(gray, (3, 3), 0))

    raw_score = (
        sharpness * 0.4 +
        contrast * 0.3 +
        brightness_score * 0.2 -
        noise * 0.1 +
        face_score * 100      # boost if face detected
    )

    return {
        "image": os.path.basename(path),
        "raw_score": raw_score,
        "sharpness": sharpness,
        "face": bool(face_score)
    }


# ---------- USER INPUT ----------
folder = input("Enter image folder path: ").strip()

if not os.path.isdir(folder):
    print("❌ Invalid folder path")
    exit()

files = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith(SUPPORTED_EXT)
]

if not files:
    print("❌ No images found")
    exit()

results = []

# ---------- MULTI-THREADED PROCESSING ----------
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(rate_image, f) for f in files]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Rating images",
        unit="img"
    ):
        res = future.result()
        if res:
            results.append(res)

if not results:
    print("❌ All images were filtered out (too blurry?)")
    exit()

# ---------- NORMALIZE SCORES (0–100) ----------
df = pd.DataFrame(results)

min_score = df["raw_score"].min()
max_score = df["raw_score"].max()

df["score"] = ((df["raw_score"] - min_score) /
               (max_score - min_score) * 100).round(2)

df = df.sort_values(by="score", ascending=False)

# ---------- FINAL OUTPUT ----------
print("\n🏆 BEST IMAGES OVERALL:\n")
print(df[["image", "score", "face"]].head(10).to_string(index=False))

df.to_csv("image_ranking.csv", index=False)
print("\n✅ Saved full results to image_ranking.csv")
