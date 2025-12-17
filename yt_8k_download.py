import os
import yt_dlp

# ---------- Progress Hook ----------
def progress_hook(d):
    if d['status'] == 'downloading':
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded_bytes = d.get('downloaded_bytes', 0)
        percentage = (downloaded_bytes / total_bytes * 100) if total_bytes else 0
        downloaded_mb = downloaded_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
        speed = d.get('speed') or 0
        speed_mb = speed / (1024 * 1024) if speed else 0
        eta = d.get('eta') or 0

        bar_length = 30
        filled_length = int(bar_length * percentage // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(
            f"\r[{bar}] {percentage:3.0f}% "
            f"{downloaded_mb:.2f}/{total_mb:.2f} MB "
            f"Speed: {speed_mb:.2f} MB/s ETA: {eta}s",
            end=''
        )

    elif d['status'] == 'finished':
        print(f"\nDownload completed: {d['filename']}")

# ---------- Inputs ----------
video_url = input("Video link: ").strip()
save_folder = input("Where to save: ").strip()
os.makedirs(save_folder, exist_ok=True)

# ---------- Get Video Info ----------
ydl_opts_info = {'quiet': True}
with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
    info = ydl.extract_info(video_url, download=False)
    title = info.get('title', 'Unknown Title')
    formats = info.get('formats', [])

resolutions = sorted(
    {f['height'] for f in formats if f.get('height')},
    reverse=True
)

# ---------- Resolution → Codec Loop ----------
while True:
    print("\nAvailable resolutions:")
    for i, res in enumerate(resolutions, 1):
        print(f"{i}: {res}p")

    res_choice = input("Choose resolution (number): ").strip()

    if not (res_choice.isdigit() and 1 <= int(res_choice) <= len(resolutions)):
        print("Invalid resolution choice. Try again.")
        continue

    desired_res = str(resolutions[int(res_choice) - 1])

    # ---------- Codec Menu ----------
    while True:
        print(f"\nSelected resolution: {desired_res}p")
        print("Select codec:")
        print("1: AV1 (best quality)")
        print("2: VP9")
        print("3: H.264")
        print("B: Back to resolution menu")

        codec_choice = input("Choose codec: ").strip().lower()

        codec_map = {
            '1': 'av1',
            '2': 'vp9',
            '3': 'avc1'
        }

        if codec_choice == 'b':
            break

        if codec_choice not in codec_map:
            print("Invalid codec choice.")
            continue

        selected_codec = codec_map[codec_choice]
        print(f"\nDownloading {title}")
        print(f"Resolution: {desired_res}p | Codec: {selected_codec.upper()}\n")

        # ---------- Download ----------
        outtmpl = os.path.join(save_folder, '%(title)s.%(ext)s')
        ydl_opts_download = {
            'format': f'bestvideo[vcodec*={selected_codec}][height={desired_res}]+bestaudio/best',
            'outtmpl': outtmpl,
            'progress_hooks': [progress_hook],
            'quiet': True,
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            ydl.download([video_url])

        exit(0)
