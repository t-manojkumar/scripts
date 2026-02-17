#!/usr/bin/env python3
"""
yt_audio_downloader.py
Interactive YouTube audio downloader with a premium terminal UI.
"""

import itertools
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from yt_dlp import YoutubeDL


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR SYSTEM
#  Auto-disabled when not a TTY (piped / redirected output)
# ══════════════════════════════════════════════════════════════════════════════

_TTY = sys.stdout.isatty()

def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _TTY else t

# Palette
def orange(t):     return _c("38;5;214", t)   # brand accent — warm amber
def gold(t):       return _c("38;5;220", t)   # secondary highlight
def smoke(t):      return _c("38;5;245", t)   # muted secondary text
def ghost(t):      return _c("38;5;238", t)   # very dim / decorative
def green(t):      return _c("38;5;120", t)   # success
def red(t):        return _c("38;5;203", t)   # error
def yellow(t):     return _c("38;5;228", t)   # warning
def white(t):      return _c("97", t)          # primary text
def bold(t):       return _c("1",  t)
def dim(t):        return _c("2",  t)
def italic(t):     return _c("3",  t)


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE FILTERS
# ══════════════════════════════════════════════════════════════════════════════

_MUTED = re.compile(
    r"(GVS PO Token|SABR streaming|Falling back|po_token"
    r"|Remote components challenge solver|Signature solving failed"
    r"|n challenge solving failed|Skipping unsupported client)",
    re.IGNORECASE,
)
_JS_RE = re.compile(
    r"(Signature solving failed|n challenge solving failed"
    r"|Remote components challenge solver)",
    re.IGNORECASE,
)
_js_warned = False


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class SilentLogger:
    def debug(self, _):   pass
    def info(self, _):    pass

    def warning(self, msg: str) -> None:
        global _js_warned
        clean = re.sub(r"^\[.*?\]\s[\w-]+:\s", "", msg).strip()
        if _JS_RE.search(clean):
            if not _js_warned:
                _js_warned = True
                _print_warning(
                    "JS challenge solver missing  "
                    + smoke("install Node.js → https://nodejs.org")
                )
            return
        if _MUTED.search(clean):
            return
        _print_warning(clean)

    def error(self, msg: str) -> None:
        clean = re.sub(r"^\[.*?\]\s[\w-]+:\s", "", msg).strip()
        # suppress "already recorded" archive messages — handled visually
        if "has already been recorded" in clean:
            return
        _print_error(clean)


# ══════════════════════════════════════════════════════════════════════════════
#  SPINNER
# ══════════════════════════════════════════════════════════════════════════════

_SPINNER_FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

class Spinner:
    def __init__(self, label: str):
        self.label   = label
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for f in itertools.cycle(_SPINNER_FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {orange(f)}  {smoke(self.label)}   ")
            sys.stdout.flush()
            time.sleep(0.08)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT HELPERS  (all output goes through here for consistent indentation)
# ══════════════════════════════════════════════════════════════════════════════

def _ln(text: str = ""):
    print(text)

def _print_rule(char: str = "─", width: int = 56):
    print("  " + ghost(char * width))

def _print_warning(msg: str):
    print(f"\n  {yellow('◆')}  {smoke(msg)}")

def _print_error(msg: str):
    print(f"\n  {red('✗')}  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  SPLASH SCREEN
# ══════════════════════════════════════════════════════════════════════════════

_LOGO = r"""
    ╦ ╦╔╦╗       ╔═╗╦ ╦╔╦╗╦╔═╗
    ╚╦╝ ║─────── ╠═╣║ ║ ║║║║ ║
     ╩  ╩        ╩ ╩╚═╝═╩╝╩╚═╝
"""

def print_splash():
    _ln()
    for line in _LOGO.strip("\n").splitlines():
        print("  " + orange(line))
    _ln()
    print(f"  {smoke('pull audio from youtube  ·  fast · clean · tagged')}")
    _ln()
    _print_rule()
    _ln()


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS  (interactive inline config)
# ══════════════════════════════════════════════════════════════════════════════

CODECS   = ["m4a", "mp3", "opus", "flac"]
DEFAULTS = {
    "codec":     "m4a",
    "quality":   "192",
    "thumbnail": True,
    "archive":   True,
}

def _codec_display(cfg: dict) -> str:
    parts = []
    for c in CODECS:
        parts.append(bold(orange(c)) if c == cfg["codec"] else smoke(c))
    return "  ·  ".join(parts)

def _bool_display(val: bool) -> str:
    return green("on") if val else smoke("off")

def print_settings(cfg: dict):
    music_dir = str(Path.home() / "Music")
    _ln()
    print(f"  {ghost('◆')}  {white('settings')}")
    _ln()
    print(f"  {'codec':<14}{_codec_display(cfg)}")
    print(f"  {'quality':<14}{smoke(cfg['quality'] + ' kbps')}")
    print(f"  {'save to':<14}{smoke(music_dir)}")
    print(f"  {'thumbnail':<14}{_bool_display(cfg['thumbnail'])}")
    print(f"  {'skip dupes':<14}{_bool_display(cfg['archive'])}")
    _ln()
    _print_rule()
    _ln()

def ask_settings(cfg: dict) -> dict:
    """Prompt the user to optionally tweak settings, or just hit enter."""
    cfg = dict(cfg)  # shallow copy

    print_settings(cfg)
    print(
        f"  {smoke('change a setting or press')} {white('enter')} "
        f"{smoke('to start downloading')}"
    )
    print(
        f"  {ghost('codec / quality / thumbnail / dupes / all defaults')}"
    )
    _ln()

    while True:
        try:
            raw = input(f"  {orange('◆')}  {white('setting')} {smoke('›')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt

        if not raw:
            break

        if raw in ("codec", "c"):
            opts = " · ".join(CODECS)
            raw2 = input(f"  {smoke('codec')} {ghost('[' + opts + ']')} {smoke('›')} ").strip().lower()
            if raw2 in CODECS:
                cfg["codec"] = raw2
        elif raw in ("quality", "q"):
            raw2 = input(f"  {smoke('quality kbps')} {ghost('[e.g. 128 / 192 / 320]')} {smoke('›')} ").strip()
            if raw2.isdigit():
                cfg["quality"] = raw2
        elif raw in ("thumbnail", "t"):
            raw2 = input(f"  {smoke('thumbnail on/off')} {smoke('›')} ").strip().lower()
            if raw2 in ("on", "yes", "y", "1"):  cfg["thumbnail"] = True
            elif raw2 in ("off", "no", "n", "0"): cfg["thumbnail"] = False
        elif raw in ("dupes", "d", "archive", "a"):
            raw2 = input(f"  {smoke('skip duplicates on/off')} {smoke('›')} ").strip().lower()
            if raw2 in ("on", "yes", "y", "1"):  cfg["archive"] = True
            elif raw2 in ("off", "no", "n", "0"): cfg["archive"] = False
        elif raw in ("defaults", "reset"):
            cfg = dict(DEFAULTS)
        else:
            print(f"  {ghost('unknown setting — try: codec / quality / thumbnail / dupes')}")
            continue

        # Refresh display after change
        print()
        print_settings(cfg)
        print(f"  {smoke('change another or press')} {white('enter')} {smoke('to start')}")
        _ln()

    _ln()
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  URL COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def collect_urls() -> list[str]:
    print(f"  {ghost('◆')}  {white('queue')}")
    _ln()
    print(f"  {smoke('paste one url per line')}")
    print(f"  {smoke('leave blank and press')} {white('enter')} {smoke('to continue')}")
    _ln()

    urls = []
    idx  = 1
    while True:
        try:
            raw = input(f"  {orange('◆')}  {white(str(idx))} {smoke('›')} ").strip()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt

        if not raw:
            if not urls:
                print(f"  {ghost('add at least one url first')}")
                continue
            break

        # Basic URL sanity check
        if not raw.startswith(("http://", "https://")):
            print(f"  {red('✗')}  {smoke('does not look like a url — try again')}")
            continue

        urls.append(raw)
        idx += 1

    _ln()
    _print_rule()
    return urls


# ══════════════════════════════════════════════════════════════════════════════
#  QUEUE DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

_STATE_ICON = {
    "pending":    smoke("○"),
    "active":     orange("◆"),
    "done":       green("✓"),
    "skipped":    smoke("◇"),
    "error":      red("✗"),
}

def print_queue(urls: list[str], states: dict, current: int):
    _ln()
    for i, url in enumerate(urls):
        state  = states.get(i, "pending")
        icon   = _STATE_ICON[state]
        # Truncate long URLs for display
        label  = url if len(url) <= 52 else url[:49] + "…"
        suffix = smoke(" · already in library") if state == "skipped" else ""
        label_col = smoke(label) if state == "pending" else (
            white(label) if state == "active" else dim(label)
        )
        print(f"  {icon}  {label_col}{suffix}")
    _ln()


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS HOOK
# ══════════════════════════════════════════════════════════════════════════════

BAR_W = 28

def _bar(pct: float) -> str:
    n = int(BAR_W * pct / 100)
    return (
        ghost("▕")
        + orange("█" * n)
        + ghost("░" * (BAR_W - n))
        + ghost("▏")
    )

# Shared state between hook and main
_current_title  = ""
_download_done  = False

def make_progress_hook():
    _done = [False]

    def hook(d: dict):
        status = d["status"]

        if status == "downloading":
            total      = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            speed      = d.get("speed")
            eta        = d.get("eta")

            spd = f"{speed/1024/1024:.1f} MB/s" if speed else "··· MB/s"
            eta_s = f"{eta}s" if eta is not None else "?s"

            if total:
                pct = downloaded * 100 / total
                print(
                    f"\r  {_bar(pct)} {orange(f'{pct:5.1f}%')}"
                    f"  {smoke(spd)}  {ghost('eta ' + eta_s)}   ",
                    end="", flush=True,
                )
            else:
                mb = downloaded / 1024 / 1024
                print(
                    f"\r  {ghost('▕' + '░' * BAR_W + '▏')}  "
                    f"{smoke(f'{mb:.1f} MB')}  {smoke(spd)}   ",
                    end="", flush=True,
                )

        elif status == "finished":
            # Snap bar to 100% then hand off to spinner for ffmpeg
            print(
                f"\r  {ghost('▕')}{orange('█' * BAR_W)}{ghost('▏')}"
                f" {orange('100.0%')}   ",
                flush=True,
            )
            _done[0] = True

        elif status == "error":
            print(f"\n  {red('✗')}  fragment error — retrying", flush=True)

    return hook, _done


# ══════════════════════════════════════════════════════════════════════════════
#  POSTPROCESSORS
# ══════════════════════════════════════════════════════════════════════════════

def build_postprocessors(codec: str, quality: str, embed_thumb: bool) -> list:
    pp = [
        {"key": "FFmpegMetadata", "add_metadata": True},
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": codec,
            "preferredquality": quality if codec != "flac" else "0",
        },
    ]
    if embed_thumb:
        pp += [
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "EmbedThumbnail"},
        ]
    return pp


# ══════════════════════════════════════════════════════════════════════════════
#  ARCHIVE
# ══════════════════════════════════════════════════════════════════════════════

def get_archive_path() -> Path:
    p = Path.home() / ".cache" / "yt-audio-dl"
    p.mkdir(parents=True, exist_ok=True)
    return p / "downloaded.txt"

def is_playlist_url(url: str) -> bool:
    return "list=" in url


# ══════════════════════════════════════════════════════════════════════════════
#  PER-TRACK DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_one(url: str, cfg: dict, archive_path) -> str:
    """
    Download a single URL.
    Returns: "done" | "skipped" | "error"
    """
    playlist    = is_playlist_url(url)
    music_dir   = Path.home() / "Music"
    music_dir.mkdir(parents=True, exist_ok=True)

    if playlist:
        outtmpl = str(music_dir / "%(playlist_title)s" / "%(playlist_index)s - %(title)s.%(ext)s")
    else:
        outtmpl = str(music_dir / "%(title)s.%(ext)s")

    progress_hook, _done = make_progress_hook()

    ydl_opts = {
        "format":            "bestaudio/best",
        "outtmpl":           outtmpl,
        "logger":            SilentLogger(),
        "restrictfilenames": True,
        "writethumbnail":    cfg["thumbnail"],
        **({"download_archive": str(archive_path)} if archive_path else {}),
        "extractor_args": {
            "youtube": {"player_client": ["tv_embedded", "android"]},
        },
        "retries":           10,
        "fragment_retries":  10,
        "concurrent_fragment_downloads": 1,
        "postprocessors":    build_postprocessors(
            cfg["codec"], cfg["quality"], cfg["thumbnail"]
        ),
        "progress_hooks":    [progress_hook],
    }

    # Fetch title first for the track header
    title = _resolve_title(url)
    if title:
        print(f"  {smoke('track')}  {white(title)}")
    _ln()

    result = "done"
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ret = ydl.download([url])
            # yt-dlp returns 0 on success, 101 when archive-skipped
            if ret == 101:
                result = "skipped"
    except SystemExit as e:
        if e.code == 101:
            result = "skipped"
        else:
            result = "error"
    except Exception:
        result = "error"

    if result == "done":
        # Brief spinner for ffmpeg encode phase (already started by yt-dlp)
        with Spinner("encoding  ·  embedding tags + artwork"):
            time.sleep(0.4)   # ffmpeg finishes async; give it a moment
        _ln()
        print(f"  {green('✓')}  {smoke('saved to')} {white(str(music_dir))}")
    elif result == "skipped":
        print(f"  {smoke('◇')}  {smoke('already in library  ·  skipped')}")
    else:
        _print_error("download failed  ·  see warnings above")

    return result


def _resolve_title(url: str) -> str:
    """Quietly extract the video title without downloading."""
    try:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "logger": SilentLogger(),
            "extractor_args": {
                "youtube": {"player_client": ["tv_embedded", "android"]},
            },
        }
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "") if info else ""
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(states: dict, total: int):
    done    = sum(1 for s in states.values() if s == "done")
    skipped = sum(1 for s in states.values() if s == "skipped")
    errors  = sum(1 for s in states.values() if s == "error")

    _print_rule()
    _ln()
    print(f"  {ghost('◆')}  {white('session complete')}")
    _ln()

    if done:
        print(f"  {green('✓')}  {bold(str(done))} {smoke('downloaded')}")
    if skipped:
        print(f"  {smoke('◇')}  {bold(str(skipped))} {smoke('already in library')}")
    if errors:
        print(f"  {red('✗')}  {bold(str(errors))} {smoke('failed')}")

    _ln()
    print(f"  {smoke('library')}  {white(str(Path.home() / 'Music'))}")
    _ln()


# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_deps():
    if shutil.which("ffmpeg") is None:
        print(
            f"\n  {red('✗')}  ffmpeg not found\n\n"
            f"  {smoke('install it first:')}\n"
            f"  {ghost('macOS')}   brew install ffmpeg\n"
            f"  {ghost('Ubuntu')}  sudo apt install ffmpeg\n"
            f"  {ghost('Win')}     https://ffmpeg.org/download.html\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    check_deps()

    try:
        print_splash()
        urls = collect_urls()
        cfg  = ask_settings(DEFAULTS)
    except KeyboardInterrupt:
        print(f"\n\n  {smoke('bye.')}\n")
        sys.exit(0)

    archive_path = get_archive_path() if cfg["archive"] else None
    states: dict[int, str] = {}

    for i, url in enumerate(urls):
        states[i] = "active"
        print_queue(urls, states, i)
        _print_rule()
        _ln()

        try:
            result = download_one(url, cfg, archive_path)
        except KeyboardInterrupt:
            print(f"\n\n  {smoke('cancelled.')}\n")
            sys.exit(0)

        states[i] = result
        _ln()
        _print_rule()
        _ln()

    print_summary(states, len(urls))


if __name__ == "__main__":
    main()