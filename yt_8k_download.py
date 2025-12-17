import os
import subprocess
import yt_dlp

# ---------- User Inputs ----------
save_folder = input("Enter folder to save videos: ").strip()
os.makedirs(save_folder, exist_ok=True)

video_url = input("Enter YouTube video link: ").strip()

# ---------- yt-dlp: Extract Best Video/Audio URL ----------
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'quiet': True,
    'noplaylist': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=False)
    formats = info.get('formats', [])

    # Extract best video
    video_formats = [f for f in formats if f.get('vcodec') != 'none']
    if not video_formats:
        raise ValueError("No video streams found.")
    best_video = max(video_formats, key=lambda x: x.get('height') or 0)

    # Extract best audio
    audio_formats = [f for f in formats if f.get('acodec') != 'none']
    if not audio_formats:
        raise ValueError("No audio streams found.")
    best_audio = max(audio_formats, key=lambda x: x.get('abr') or 0)

    # For maximum speed, use the video URL (usually combined video+audio if available)
    direct_url = best_video.get('url')
    file_name = ydl.prepare_filename(info)

# ---------- Save Direct URL to Temporary File ----------
temp_url_file = os.path.join(save_folder, "aria2_url.txt")
with open(temp_url_file, 'w') as f:
    f.write(direct_url + '\n')

# ---------- Download with aria2 ----------
print(f"\nDownloading '{file_name}' with maximum speed using aria2...")

aria2_cmd = [
    "aria2c",
    "-i", temp_url_file,        # Input URL file
    "-d", save_folder,          # Download folder
    "-x", "16",                 # Max connections per server
    "-s", "16",                 # Split file into 16 segments
    "--continue=true",          # Resume incomplete downloads
    "--auto-file-renaming=false",
]

try:
    subprocess.run(aria2_cmd, check=True)
    print(f"\nDownload completed: {file_name}")
except subprocess.CalledProcessError as e:
    print(f"\nDownload failed: {e}")
