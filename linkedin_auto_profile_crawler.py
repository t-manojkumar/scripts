"""
LinkedIn profile scraper using StaffSpy + Chrome Profile 1 session.

Install:
    pip install -U "staffspy[browser]" pymongo pywin32 cryptography

Run (uses your existing Chrome Profile 1 LinkedIn login — no manual login needed):
    python linkedin_auto_profile_crawler.py https://www.linkedin.com/in/-sriram-t/ ^
        --chrome-profile-path "C:\\Users\\MANO\\AppData\\Local\\Google\\Chrome\\User Data\\Profile 1"

On first run it extracts cookies from Profile 1 and saves them to staffspy_session.pkl.
Subsequent runs reuse the pickle. Re-run with --refresh-session when LinkedIn signs you out.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


SESSION_FILE  = Path("staffspy_session.pkl")
OUTPUT_DIR    = Path("linkedin_output")
MONGO_URI     = "mongodb://localhost:27017"
MONGO_DB      = "linkedin"
MONGO_COLLECTION = "profiles"

DEFAULT_CHROME_PROFILE = Path(os.environ.get("LOCALAPPDATA", "")) / \
    "Google" / "Chrome" / "User Data" / "Profile 1"

# LinkedIn cookies required for API access
_LI_COOKIE_NAMES = {"li_at", "JSESSIONID", "liap", "li_a", "lidc", "bcookie"}

_SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "X-RestLi-Protocol-Version": "2.0.0",
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape a LinkedIn profile with StaffSpy using Chrome Profile 1 session."
    )
    parser.add_argument(
        "profile_url",
        help="LinkedIn profile URL, e.g. https://www.linkedin.com/in/name/",
    )
    parser.add_argument(
        "--chrome-profile-path",
        default=str(DEFAULT_CHROME_PROFILE),
        help="Path to your Chrome profile directory (contains Cookies file).",
    )
    parser.add_argument(
        "--session-file",
        default=str(SESSION_FILE),
        help="Where to store/load the StaffSpy session pickle.",
    )
    parser.add_argument(
        "--refresh-session",
        action="store_true",
        help="Force re-extract cookies from Chrome even if session file exists.",
    )
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--mongo-db", default=MONGO_DB)
    parser.add_argument("--mongo-collection", default=MONGO_COLLECTION)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-mongo", action="store_true", help="Skip MongoDB.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# URL helpers
# ──────────────────────────────────────────────────────────────────────────────


def extract_user_id(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parts = [p for p in urlparse(url).path.split("/") if p]
    if "in" in parts:
        idx = parts.index("in")
        if idx + 1 < len(parts):
            return parts[idx + 1].rstrip("/")
    raise ValueError(f"Cannot extract user ID from: {url}")


# ──────────────────────────────────────────────────────────────────────────────
# Chrome cookie extraction  (Windows only — requires pywin32 + cryptography)
# ──────────────────────────────────────────────────────────────────────────────


def _get_aes_key(local_state_path: Path) -> bytes:
    """Decrypt Chrome's AES cookie key stored in Local State via Windows DPAPI."""
    try:
        import win32crypt  # noqa: F401 — tested below
    except ImportError:
        raise RuntimeError(
            "pywin32 is required to read Chrome cookies.\n"
            "Run:  pip install pywin32"
        )

    with open(local_state_path, encoding="utf-8") as f:
        local_state = json.load(f)

    enc_key_b64: str = local_state["os_crypt"]["encrypted_key"]
    enc_key = base64.b64decode(enc_key_b64)[5:]          # strip "DPAPI" prefix
    import win32crypt
    return win32crypt.CryptUnprotectData(enc_key, None, None, None, 0)[1]


def _decrypt_value(encrypted: bytes, aes_key: bytes) -> str:
    if not encrypted:
        return ""
    # Chrome v80+ — AES-256-GCM, prefixed with b'v10' or b'v11'
    if encrypted[:3] in (b"v10", b"v11"):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            nonce      = encrypted[3:15]
            ciphertext = encrypted[15:]
            return AESGCM(aes_key).decrypt(nonce, ciphertext, None).decode("utf-8")
        except Exception:
            return ""
    # Older Chrome — plain DPAPI
    try:
        import win32crypt
        return win32crypt.CryptUnprotectData(encrypted, None, None, None, 0)[1].decode("utf-8")
    except Exception:
        return ""


def _copy_locked(src: Path, dst: Path) -> None:
    """
    Copy a file Chrome has open, using Win32 CreateFileW with full sharing flags.
    Falls back through three methods so at least one succeeds.
    """
    import ctypes
    import ctypes.wintypes as wt

    GENERIC_READ         = 0x80000000
    FILE_SHARE_ALL       = 0x00000007   # READ | WRITE | DELETE
    OPEN_EXISTING        = 3
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    k32    = ctypes.windll.kernel32
    handle = k32.CreateFileW(
        str(src), GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING, 0, None,
    )

    if handle not in (INVALID_HANDLE_VALUE, 0, None):
        try:
            # Use Python stat() for file size — avoids GetFileSize sign issues
            size = src.stat().st_size
            if size > 0:
                buf  = ctypes.create_string_buffer(size)
                read = wt.DWORD(0)
                k32.ReadFile(handle, buf, size, ctypes.byref(read), None)
                dst.write_bytes(bytes(buf)[: read.value])
                return
        finally:
            k32.CloseHandle(handle)

    # Method 2: sqlite3 immutable flag bypasses SQLite's own lock checks
    # (works when Chrome allows shared OS reads but shutil still fails)
    try:
        conn = sqlite3.connect(f"file:{src}?mode=ro&immutable=1", uri=True)
        raw  = conn.execute("pragma integrity_check").fetchone()  # force full read
        conn.close()
        # Re-open and dump via sqlite3 backup API
        src_conn = sqlite3.connect(f"file:{src}?mode=ro&immutable=1", uri=True)
        dst_conn = sqlite3.connect(str(dst))
        src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
        return
    except Exception:
        pass

    # Method 3: plain copy (Chrome may have been closed by now)
    shutil.copy2(src, dst)


def extract_chrome_linkedin_cookies(profile_dir: Path) -> dict[str, str]:
    """
    Read LinkedIn session cookies from Chrome Profile 1's SQLite Cookies DB.
    Chrome can stay open — we open the file with Win32 sharing flags so
    Chrome's exclusive handle doesn't block us.
    """
    profile_dir = Path(profile_dir)
    local_state = profile_dir.parent / "Local State"

    if not local_state.exists():
        raise RuntimeError(f"Local State not found: {local_state}")

    aes_key = _get_aes_key(local_state)

    # Chrome v96+ → Network/Cookies; older → Cookies
    cookie_src: Path | None = None
    for candidate in [profile_dir / "Network" / "Cookies", profile_dir / "Cookies"]:
        if candidate.exists():
            cookie_src = candidate
            break

    if cookie_src is None:
        raise RuntimeError(
            f"Chrome Cookies file not found under {profile_dir}.\n"
            "Visit linkedin.com in Chrome and log in first."
        )

    print(f"  Reading cookie DB: {cookie_src}", flush=True)

    # Write to a temp file via Win32 (bypasses Chrome's file lock)
    tmp = Path(tempfile.mktemp(suffix=".db"))
    try:
        _copy_locked(cookie_src, tmp)

        conn = sqlite3.connect(f"file:{tmp}?mode=ro&immutable=1", uri=True)
        # Fetch ALL linkedin.com cookies so we can show what's available
        all_rows = conn.execute(
            "SELECT name, encrypted_value FROM cookies "
            "WHERE host_key LIKE '%linkedin.com%'"
        ).fetchall()
        conn.close()
    finally:
        tmp.unlink(missing_ok=True)

    print(f"  LinkedIn cookies found: {[r[0] for r in all_rows]}", flush=True)

    cookies: dict[str, str] = {}
    for name, enc_val in all_rows:
        if name not in _LI_COOKIE_NAMES:
            continue
        val = _decrypt_value(enc_val, aes_key)
        if val:
            cookies[name] = val

    if "li_at" not in cookies:
        raise RuntimeError(
            f"LinkedIn 'li_at' cookie not found. "
            f"Cookies present: {[r[0] for r in all_rows]}\n"
            "Make sure you are logged into LinkedIn in Chrome Profile 1 "
            "(visit linkedin.com and check you see your feed)."
        )

    return cookies


# ──────────────────────────────────────────────────────────────────────────────
# StaffSpy session file builder
# ──────────────────────────────────────────────────────────────────────────────


def build_session_file(cookies: dict[str, str], session_file: Path) -> None:
    """
    Create a StaffSpy-compatible session pickle from LinkedIn cookies.
    StaffSpy stores: {"cookies": RequestsCookieJar, "headers": dict}
    """
    import requests

    session = requests.Session()
    session.headers.update(_SESSION_HEADERS)
    for name, value in cookies.items():
        session.cookies.set(name, value, domain=".linkedin.com", path="/")

    data = {"cookies": session.cookies, "headers": dict(session.headers)}
    session_file.parent.mkdir(parents=True, exist_ok=True)
    with open(session_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Session saved  : {session_file}  ({len(cookies)} cookies)", flush=True)
    print(f"  Cookies      : {', '.join(cookies)}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame → dict
# ──────────────────────────────────────────────────────────────────────────────


def df_row_to_dict(df: "pd.DataFrame") -> dict:  # type: ignore[name-defined]
    import numpy as np
    import pandas as pd

    row = df.iloc[0].to_dict()
    out: dict = {}
    for key, val in row.items():
        if isinstance(val, float) and np.isnan(val):
            out[key] = None
        elif isinstance(val, np.integer):
            out[key] = int(val)
        elif isinstance(val, np.floating):
            out[key] = float(val)
        elif isinstance(val, np.bool_):
            out[key] = bool(val)
        elif isinstance(val, pd.Timestamp):
            out[key] = val.isoformat()
        else:
            out[key] = val
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────


def save_json(record: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = str(record.get("profile_id") or record.get("name") or "profile")
    slug = slug.replace(" ", "_").replace("/", "_")
    path = out_dir / f"{slug}.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def save_mongo(record: dict, uri: str, db_name: str, collection_name: str) -> None:
    try:
        from pymongo import ASCENDING, MongoClient
    except ImportError:
        print("MongoDB skipped — run:  pip install pymongo")
        return

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        col = client[db_name][collection_name]

        col.create_index([("profile_id", ASCENDING)], unique=True, background=True)
        col.create_index([("profile_link", ASCENDING)], background=True)
        col.create_index([("fetched_at", ASCENDING)], background=True)

        result = col.update_one(
            {"profile_id": record["profile_id"]},
            {"$set": record},
            upsert=True,
        )
        client.close()
        action = "inserted" if result.upserted_id else "updated"
        print(f"MongoDB {action}: {db_name}.{collection_name} → {record.get('profile_id')}")
    except Exception as exc:
        print(f"MongoDB skipped: {type(exc).__name__}: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    try:
        from staffspy import LinkedInAccount
    except ImportError:
        print(
            'StaffSpy not installed. Run:\n'
            '  pip install -U "staffspy[browser]" pywin32 cryptography',
            file=sys.stderr,
        )
        return 1

    session_file = Path(args.session_file)
    profile_dir  = Path(args.chrome_profile_path)

    # Build / refresh session from Chrome Profile 1
    if not session_file.exists() or args.refresh_session:
        print(f"Extracting LinkedIn cookies from: {profile_dir}", flush=True)
        cookies = extract_chrome_linkedin_cookies(profile_dir)
        build_session_file(cookies, session_file)
    else:
        print(f"Reusing session: {session_file}", flush=True)

    user_id = extract_user_id(args.profile_url)
    print(f"Target profile : {user_id}", flush=True)

    account = LinkedInAccount(
        session_file=str(session_file),
        log_level=1,
    )

    print("Scraping…", flush=True)
    df = account.scrape_users(user_ids=[user_id])

    if df is None or df.empty:
        print(
            "No data returned. The profile may be private, or the session has expired.\n"
            "Re-run with --refresh-session to pull fresh cookies from Chrome."
        )
        return 1

    record = df_row_to_dict(df)
    record["source_url"] = args.profile_url
    record["fetched_at"] = datetime.now(timezone.utc).isoformat()
    record["scraper"]    = "staffspy"

    out_path = save_json(record, Path(args.out_dir))
    print(f"JSON saved     : {out_path.resolve()}")
    print(f"Fields         : {[k for k, v in record.items() if v is not None]}")

    if not args.no_mongo:
        save_mongo(record, args.mongo_uri, args.mongo_db, args.mongo_collection)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
