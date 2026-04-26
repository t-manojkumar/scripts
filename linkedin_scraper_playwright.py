"""
Fetch visible LinkedIn profile or company page data with Playwright.

This script is intended for pages you are allowed to access. It supports a
manual login flow and reuses the saved browser session on later runs. It does
not bypass captchas, paywalls, privacy controls, or LinkedIn access limits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import BrowserContext, Page, TimeoutError, sync_playwright


DEFAULT_STATE_PATH = Path("linkedin_auth_state.json")
DEFAULT_OUTPUT_PATH = Path("linkedin_data.json")
SECTION_HEADINGS = {
    "About",
    "Experience",
    "Education",
    "Licenses & certifications",
    "Licenses and certifications",
    "Skills",
    "Recommendations",
    "Projects",
    "Publications",
    "Volunteer experience",
    "Activity",
    "Posts",
    "Contact info",
    "People",
    "Jobs",
    "Overview",
    "Locations",
    "Employees",
    "Updates",
    "Similar pages",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape visible LinkedIn page data into JSON and optionally MongoDB."
    )
    parser.add_argument("urls", nargs="*", help="LinkedIn profile/company URLs to fetch.")
    parser.add_argument("--urls-file", help="Text file with one LinkedIn URL per line.")
    parser.add_argument("--json-out", default=str(DEFAULT_OUTPUT_PATH), help="JSON output path.")
    parser.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help="Playwright storage state file for your LinkedIn session.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Open a headed browser for manual LinkedIn login and save session state.",
    )
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument(
        "--delay-min",
        type=float,
        default=2.0,
        help="Minimum delay between pages in seconds.",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=5.0,
        help="Maximum delay between pages in seconds.",
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"), help="MongoDB URI.")
    parser.add_argument("--mongo-db", default="linkedin", help="MongoDB database name.")
    parser.add_argument(
        "--mongo-collection",
        default="pages",
        help="MongoDB collection name.",
    )
    parser.add_argument(
        "--upsert-mongo",
        action="store_true",
        help="Upsert MongoDB documents by source_url instead of inserting every run.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=45000,
        help="Navigation and selector timeout in milliseconds.",
    )
    return parser.parse_args()


def load_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = list(args.urls)
    if args.urls_file:
        path = Path(args.urls_file)
        urls.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    normalized = []
    seen = set()
    for url in urls:
        clean = normalize_linkedin_url(url)
        if clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


def normalize_linkedin_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("Empty URL provided.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "linkedin.com" not in host:
        raise ValueError(f"Not a LinkedIn URL: {url}")
    return url


def polite_sleep(min_seconds: float, max_seconds: float) -> None:
    if max_seconds <= 0:
        return
    time.sleep(random.uniform(max(0, min_seconds), max(min_seconds, max_seconds)))


def create_context(browser: Any, state_path: Path) -> BrowserContext:
    kwargs: dict[str, Any] = {
        "viewport": {"width": 1366, "height": 900},
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
    }
    if state_path.exists():
        kwargs["storage_state"] = str(state_path)
    return browser.new_context(**kwargs)


def ensure_login(context: BrowserContext, state_path: Path, timeout_ms: int) -> None:
    page = context.new_page()
    page.set_default_timeout(timeout_ms)
    page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
    print("\nA browser window is open for LinkedIn login.")
    print("Log in manually. If LinkedIn asks for verification, complete it in the browser.")
    input("After the feed or target account is visible, press Enter here to save the session...")
    context.storage_state(path=str(state_path))
    page.close()
    print(f"Saved LinkedIn session to: {state_path.resolve()}")


def fetch_page(page: Page, url: str, timeout_ms: int) -> dict[str, Any]:
    page.set_default_timeout(timeout_ms)
    response_status = None
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        response_status = response.status if response else None
        wait_for_page_ready(page)
    except TimeoutError:
        pass

    page_type = detect_page_type(page.url)
    text_lines = visible_text_lines(page)
    meta = extract_meta(page)
    sections = extract_sections(text_lines)

    data: dict[str, Any] = {
        "source_url": url,
        "final_url": page.url,
        "page_type": page_type,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "http_status": response_status,
        "title": safe_title(page),
        "meta": meta,
        "primary": extract_primary_fields(page, page_type, meta, text_lines),
        "sections": sections,
        "links": extract_links(page),
        "text_lines": text_lines,
    }
    return data


def wait_for_page_ready(page: Page) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except TimeoutError:
        pass
    try:
        page.locator("main").first.wait_for(state="visible", timeout=15000)
    except TimeoutError:
        pass

    # A gentle scroll wakes lazy-loaded sections without simulating aggressive crawling.
    for _ in range(4):
        page.mouse.wheel(0, 1200)
        time.sleep(0.8)
    page.mouse.wheel(0, -5000)
    time.sleep(0.5)


def detect_page_type(url: str) -> str:
    path = urlparse(url).path.lower()
    if "/in/" in path:
        return "profile"
    if "/company/" in path or "/school/" in path:
        return "organization"
    if "/jobs/" in path:
        return "job"
    return "linkedin_page"


def extract_meta(page: Page) -> dict[str, str | None]:
    selectors = {
        "description": "meta[name='description']",
        "og_title": "meta[property='og:title']",
        "og_description": "meta[property='og:description']",
        "og_image": "meta[property='og:image']",
        "canonical": "link[rel='canonical']",
    }
    meta: dict[str, str | None] = {}
    for key, selector in selectors.items():
        attr = "href" if key == "canonical" else "content"
        meta[key] = page.locator(selector).first.get_attribute(attr)
    return meta


def safe_title(page: Page) -> str | None:
    try:
        return page.title()
    except Exception:
        return None


def visible_text_lines(page: Page) -> list[str]:
    text = page.locator("main").first.inner_text(timeout=10000) if page.locator("main").count() else page.locator("body").inner_text()
    lines = []
    seen = set()
    for line in text.splitlines():
        cleaned = normalize_space(line)
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        lines.append(cleaned)
        seen.add(cleaned)
    return lines


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_primary_fields(
    page: Page,
    page_type: str,
    meta: dict[str, str | None],
    text_lines: list[str],
) -> dict[str, Any]:
    h1 = first_inner_text(page, "h1")
    primary: dict[str, Any] = {
        "name": h1 or clean_linkedin_title(meta.get("og_title") or ""),
        "headline": None,
        "location": None,
    }

    if page_type == "profile":
        primary.update(extract_profile_hints(primary["name"], text_lines, meta))
    elif page_type == "organization":
        primary.update(extract_organization_hints(primary["name"], text_lines, meta))
    else:
        primary["description"] = meta.get("description") or meta.get("og_description")

    return primary


def first_inner_text(page: Page, selector: str) -> str | None:
    try:
        locator = page.locator(selector).first
        if locator.count() == 0:
            return None
        return normalize_space(locator.inner_text(timeout=3000))
    except Exception:
        return None


def clean_linkedin_title(value: str) -> str | None:
    value = normalize_space(value)
    if not value:
        return None
    return re.sub(r"\s+\|\s+LinkedIn.*$", "", value, flags=re.IGNORECASE)


def extract_profile_hints(
    name: str | None,
    text_lines: list[str],
    meta: dict[str, str | None],
) -> dict[str, Any]:
    hints: dict[str, Any] = {
        "headline": None,
        "location": None,
        "connection_degree": None,
        "followers_or_connections": None,
        "description": meta.get("og_description") or meta.get("description"),
    }

    if not name:
        return hints

    try:
        idx = text_lines.index(name)
    except ValueError:
        idx = -1

    candidates = text_lines[idx + 1 : idx + 8] if idx >= 0 else text_lines[:8]
    skip_words = {"1st", "2nd", "3rd", "Contact info", "Message", "Connect", "Follow"}
    for line in candidates:
        if line in skip_words or line.startswith("More actions"):
            continue
        if hints["headline"] is None and not looks_like_location(line):
            hints["headline"] = line
            continue
        if hints["location"] is None and looks_like_location(line):
            hints["location"] = line

    for line in text_lines[:40]:
        if line in {"1st", "2nd", "3rd"}:
            hints["connection_degree"] = line
        if re.search(r"\b(followers|connections)\b", line, re.IGNORECASE):
            hints["followers_or_connections"] = line
            break
    return hints


def extract_organization_hints(
    name: str | None,
    text_lines: list[str],
    meta: dict[str, str | None],
) -> dict[str, Any]:
    hints: dict[str, Any] = {
        "tagline": None,
        "industry": None,
        "company_size": None,
        "headquarters": None,
        "description": meta.get("og_description") or meta.get("description"),
    }
    if name and name in text_lines:
        idx = text_lines.index(name)
        if idx + 1 < len(text_lines):
            hints["tagline"] = text_lines[idx + 1]

    for line in text_lines:
        lower = line.lower()
        if "employees" in lower and hints["company_size"] is None:
            hints["company_size"] = line
        elif "headquarters" in lower and hints["headquarters"] is None:
            hints["headquarters"] = line
        elif "industry" in lower and hints["industry"] is None:
            hints["industry"] = line
    return hints


def looks_like_location(line: str) -> bool:
    if "," in line and len(line) <= 80:
        return True
    location_words = ("India", "United States", "United Kingdom", "Remote", "Area")
    return any(word.lower() in line.lower() for word in location_words)


def extract_sections(text_lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    headings_lower = {heading.lower(): heading for heading in SECTION_HEADINGS}

    for line in text_lines:
        canonical = headings_lower.get(line.lower())
        if canonical:
            current = canonical
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)

    return {key: values for key, values in sections.items() if values}


def extract_links(page: Page) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    anchors = page.locator("main a[href]")
    count = min(anchors.count(), 250)
    seen = set()
    for index in range(count):
        anchor = anchors.nth(index)
        href = anchor.get_attribute("href")
        if not href or href.startswith(("javascript:", "#")):
            continue
        text = normalize_space(anchor.inner_text(timeout=1000) or "")
        key = (href, text)
        if key in seen:
            continue
        seen.add(key)
        links.append({"text": text, "href": href})
    return links


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def write_mongo(
    mongo_uri: str | None,
    db_name: str,
    collection_name: str,
    records: list[dict[str, Any]],
    upsert: bool,
) -> None:
    if not mongo_uri:
        return
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("Install pymongo to use --mongo-uri: pip install pymongo") from exc

    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]
    if upsert:
        for record in records:
            collection.update_one(
                {"source_url": record["source_url"]},
                {"$set": record},
                upsert=True,
            )
    elif records:
        collection.insert_many(records)
    client.close()


def main() -> int:
    args = parse_args()
    urls = load_urls(args)
    state_path = Path(args.state)
    output_path = Path(args.json_out)

    if not urls and not args.login:
        print("Provide at least one LinkedIn URL, or run with --login first.", file=sys.stderr)
        return 2

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=args.headless and not args.login)
        context = create_context(browser, state_path)

        if args.login:
            ensure_login(context, state_path, args.timeout_ms)

        records = []
        if urls:
            page = context.new_page()
            for index, url in enumerate(urls, start=1):
                print(f"[{index}/{len(urls)}] Fetching {url}")
                try:
                    records.append(fetch_page(page, url, args.timeout_ms))
                except Exception as exc:
                    records.append(
                        {
                            "source_url": url,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                polite_sleep(args.delay_min, args.delay_max)
            page.close()

        context.close()
        browser.close()

    if records:
        write_json(output_path, records)
        write_mongo(
            args.mongo_uri,
            args.mongo_db,
            args.mongo_collection,
            records,
            args.upsert_mongo,
        )
        print(f"Wrote {len(records)} record(s) to {output_path.resolve()}")
        if args.mongo_uri:
            print(f"Stored MongoDB records in {args.mongo_db}.{args.mongo_collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
