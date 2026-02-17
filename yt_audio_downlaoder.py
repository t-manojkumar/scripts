#!/usr/bin/env python3
"""
yt_audio_downloader.py
A robust YouTube audio downloader built on yt-dlp.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

from yt_dlp import YoutubeDL


# ─────────────────────────── ANSI colours ────────────────────────────────────
# Automatically disabled when stdout is not a TTY (e.g. piped to a file).

_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)
def red(t):    return _c("31", t)


# ─────────────────────────── Known noisy warnings ────────────────────────────
# These are YouTube-side limitations — not actionable by the user.
# Silenced to keep output clean.

_MUTED_PATTERNS = [
    r"android client https formats require a GVS PO Token",
    r"Some web client https formats have been skipped.*SABR",
    r"Falling back to.*player",
    r"po_token",
]
_MUTED_RE = re.compile("|".join(_MUTED_PATTERNS), re.IGNORECASE)


# ─────────────────────────── Dependency check ────────────────────────────────

def check_dependencies() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            red("  ✗ ffmpeg not found in PATH.\n") +
            "  Install it first:\n"
            f"    macOS   →  brew install ffmpeg\n"
            f"    Ubuntu  →  sudo apt install ffmpeg\n"
            f"    Windows →  https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)


# ─────────────────────────── Custom logger ───────────────────────────────────

class QuietLogger:
    """Pass warnings/errors to stderr; suppress everything else."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        clean = re.sub(r"^\[.*?\]\s[\w-]+:\s", "", msg).strip()
        if _MUTED_RE.search(clean):
            return
        print(f"\n  {yellow('⚠')}  {dim(clean)}", file=sys.stderr)

    def error(self, msg: str) -> None:
        clean = re.sub(r"^\[.*?\]\s[\w-]+:\s", "", msg).strip()
        print(f"\n  {red('✗')}  {clean}", file=sys.stderr)


# ─────────────────────────── Progress hook ───────────────────────────────────

BAR_WIDTH = 26

def _bar(percent: float) -> str:
    filled = int(BAR_WIDTH * percent / 100)
    return "[" + green("█" * filled) + dim("░" * (BAR_WIDTH - filled)) + "]"

def progress_hook(d: dict) -> None:
    status = d["status"]

    if status == "downloading":
        total      = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes", 0)
        speed      = d.get("speed")
        eta        = d.get("eta")

        speed_str = f"{speed / 1024 / 1024:.1f} MB/s" if speed else "--.- MB/s"
        eta_str   = f"ETA {eta}s" if eta is not None else "ETA ?s"

        if total:
            percent = downloaded * 100 / total
            print(
                f"\r  {_bar(percent)} {green(f'{percent:5.1f}%')}  "
                f"{cyan(speed_str)}  {dim(eta_str)}   ",
                end="", flush=True,
            )
        else:
            mb = downloaded / 1024 / 1024
            print(
                f"\r  {cyan(f'{mb:.2f} MB')} downloaded  {cyan(speed_str)}   ",
                end="", flush=True,
            )

    elif status == "finished":
        full_bar = "[" + green("█" * BAR_WIDTH) + "]"
        print(
            f"\r  {full_bar} {green('100.0%')}  {dim('Processing…')}   ",
            flush=True,
        )

    elif status == "error":
        print(f"\n  {red('✗')}  Fragment error — yt-dlp will retry.", file=sys.stderr)


# ─────────────────────────── Duplicate handling ──────────────────────────────

def get_archive_path() -> Path:
    """Persistent per-user archive so already-downloaded tracks are skipped."""
    cache = Path.home() / ".cache" / "yt-audio-dl"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "downloaded.txt"


# ─────────────────────────── Output template ─────────────────────────────────

def build_output_template(custom, is_playlist: bool) -> str:
    if custom:
        return custom
    music_dir = Path.home() / "Music"
    music_dir.mkdir(parents=True, exist_ok=True)
    if is_playlist:
        return str(music_dir / "%(playlist_title)s" / "%(playlist_index)s - %(title)s.%(ext)s")
    return str(music_dir / "%(title)s.%(ext)s")


def is_playlist_url(url: str) -> bool:
    return "list=" in url


# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube videos or playlists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("url", help="YouTube video or playlist URL")
    parser.add_argument("-o", "--output", default=None,
        help="Custom yt-dlp output template (overrides ~/Music)")
    parser.add_argument("-c", "--codec", default="m4a",
        choices=["m4a", "mp3", "opus", "flac"], help="Audio codec")
    parser.add_argument("-q", "--quality", default="192",
        help="Bitrate in kbps for lossy codecs. Use '0' for VBR best (mp3).")
    parser.add_argument("-p", "--parallel", type=int, default=1,
        help="Parallel fragment downloads (DASH/HLS only)")
    parser.add_argument("--no-thumbnail", action="store_true",
        help="Skip thumbnail embedding")
    parser.add_argument("--no-archive", action="store_true",
        help="Disable duplicate detection — re-download already saved tracks")
    return parser.parse_args()


# ─────────────────────────── Postprocessors ──────────────────────────────────

def build_postprocessors(codec: str, quality: str, embed_thumbnail: bool) -> list:
    pp = [
        {"key": "FFmpegMetadata", "add_metadata": True},
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": codec,
            "preferredquality": quality if codec != "flac" else "0",
        },
    ]
    if embed_thumbnail:
        pp += [
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "EmbedThumbnail"},
        ]
    return pp


# ─────────────────────────── Header ──────────────────────────────────────────

def print_header(args, outtmpl: str, embed_thumb: bool, archive_path) -> None:
    W = 52
    br  = bold(cyan("│"))
    print(bold(cyan("┌" + "─" * W + "┐")))
    title = bold("  🎵  yt-audio-dl")
    print(f"{br}{title:^{W}}{br}")
    print(bold(cyan("├" + "─" * W + "┤")))

    out_folder = str(Path(outtmpl).parent)
    rows = [
        ("Codec",      f"{bold(args.codec.upper())}  {dim(args.quality + ' kbps')}"),
        ("Thumbnail",  green("embedded") if embed_thumb else dim("skipped")),
        ("Duplicates", dim("skipped via archive") if archive_path else yellow("allowed")),
        ("Save to",    dim(out_folder)),
    ]
    for label, value in rows:
        content = f"  {cyan(label):<22}{value}"
        print(f"{br}{content}")

    print(bold(cyan("└" + "─" * W + "┘")))
    print()


# ─────────────────────────── Main ────────────────────────────────────────────

def main() -> None:
    check_dependencies()
    args = parse_args()

    playlist     = is_playlist_url(args.url)
    outtmpl      = build_output_template(args.output, playlist)
    embed_thumb  = not args.no_thumbnail
    archive_path = None if args.no_archive else get_archive_path()
    postprocessors = build_postprocessors(args.codec, args.quality, embed_thumb)

    print_header(args, outtmpl, embed_thumb, archive_path)

    if playlist:
        print(f"  {cyan('ℹ')}  Playlist detected — grouping tracks into a subfolder.\n")

    ydl_opts = {
        "format":            "bestaudio/best",
        "outtmpl":           outtmpl,
        "logger":            QuietLogger(),
        "restrictfilenames": True,

        # Duplicate handling
        **({"download_archive": str(archive_path)} if archive_path else {}),

        # Thumbnail — EmbedThumbnail postprocessor handles embedding
        # Do NOT also set "embedthumbnail": True or it double-embeds
        "writethumbnail": embed_thumb,

        # Use ios client: avoids both PO-Token and SABR warnings
        "extractor_args": {
            "youtube": {"player_client": ["ios", "web"]},
        },

        "retries":          10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": args.parallel,
        "postprocessors":   postprocessors,
        "progress_hooks":   [progress_hook],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.url])
    except KeyboardInterrupt:
        print(f"\n\n  {yellow('⚠')}  Cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n  {red('✗')}  Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  {green('✓')}  {bold('All done!')}\n")


if __name__ == "__main__":
    main()