import os
import subprocess
import yt_dlp

# ---------- User Inputs ----------
save_folder = input("Enter folder to save videos: ").strip()
os.makedirs(save_folder, exist_ok=True)

video_url = input("Enter YouTube video link: ").strip()

# ---------- Extract Best Video URL ----------
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'quiet': True,
    'noplaylist': True,
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=False)
    # Use the direct URL for the best combined video+audio
    if 'url' in info:
        direct_url = info['url']
        file_name = ydl.prepare_filename(info)
    else:
        # Fallback: pick best video+audio streams
        formats = info.get('formats', [])
        best_video = max([f for f in formats if f.get('vcodec') != 'none'], key=lambda x: x.get('height', 0))
        best_audio = max([f for f in formats if f.get('acodec') != 'none'], key=lambda x: f.get('abr', 0))
        direct_url = best_video['url']
        file_name = ydl.prepare_filename(info)

# ---------- Save Direct URL to Temporary File ----------
temp_url_file = os.path.join(save_folder, "aria2_url.txt")
with open(temp_url_file, 'w') as f:
    f.write(direct_url + '\n')

# ---------- Download with aria2 ----------
print(f"\nDownloading '{file_name}' with maximum speed...")

aria2_cmd = [
    "aria2c",
    "-i", temp_url_file,       # Input URL file
    "-d", save_folder,         # Download folder
    "-x", "16",                # Max connections per server
    "-s", "16",                # Split file into 16 segments
    "--continue=true",         # Resume incomplete downloads
    "--auto-file-renaming=false",
]

subprocess.run(aria2_cmd)
print(f"\nDownload completed: {file_name}")
