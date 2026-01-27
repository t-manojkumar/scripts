import os
import yt_dlp

def get_video_info(url: str) -> dict:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {
            "node": {}
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["android_vr"]
            }
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    formats = info.get("formats", [])

    resolutions = set()
    codec_map = {}

    for f in formats:
        height = f.get("height")
        vcodec = f.get("vcodec")

        if not height:
            continue

        resolutions.add(height)

        if vcodec and vcodec != "none":
            base_codec = vcodec.split(".")[0]
            codec_map.setdefault(str(height), set()).add(base_codec)

    return {
        "title": info.get("title", "Unknown"),
        "resolutions": sorted([str(r) for r in resolutions], reverse=True),
        "codecs": {
            h: sorted(list(c)) for h, c in codec_map.items()
        }
    }

def download_video(
    url: str,
    save_path: str,
    resolution: str,
    codec: str,
    progress_cb
):
    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
            downloaded = d.get("downloaded_bytes", 0)
            progress_cb({
                "percent": int(downloaded / total * 100),
                "speed": d.get("speed", 0),
                "eta": d.get("eta", 0)
            })
        elif d["status"] == "finished":
            progress_cb({"finished": True, "filename": d["filename"]})

    opts = {
        "format": f"bestvideo[vcodec*={codec}][height={resolution}]+bestaudio/best",
        "outtmpl": os.path.join(save_path, "%(title)s.%(ext)s"),
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "js_runtimes": {
            "node": {}
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["android_vr"]
            }
        }
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
