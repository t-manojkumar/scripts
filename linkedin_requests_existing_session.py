import argparse
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    import browser_cookie3
except ImportError:
    browser_cookie3 = None


DEFAULT_URL = "https://www.linkedin.com/in/-sriram-t/"
DEFAULT_OUT_FILE = "sriram_linkedin_requests.json"
DEFAULT_STATE_FILE = "linkedin_auth_state.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch a LinkedIn page with requests using an existing session."
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--out", default=DEFAULT_OUT_FILE)
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE_FILE,
        help="Playwright storage state JSON created by linkedin_scraper_playwright.py --login.",
    )
    parser.add_argument(
        "--browser",
        choices=["chrome", "edge", "firefox"],
        help="Optional fallback: load cookies directly from this browser.",
    )
    parser.add_argument("--mongo-uri", help="Optional MongoDB URI, e.g. mongodb://localhost:27017")
    parser.add_argument("--mongo-db", default="linkedin")
    parser.add_argument("--mongo-collection", default="profiles")
    return parser.parse_args()


def cookies_from_playwright_state(path):
    state_path = Path(path)
    if not state_path.exists():
        return None

    state = json.loads(state_path.read_text(encoding="utf-8"))
    session = requests.Session()

    for cookie in state.get("cookies", []):
        domain = cookie.get("domain", "")
        if "linkedin.com" not in domain:
            continue
        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=domain,
            path=cookie.get("path", "/"),
        )

    if not session.cookies:
        raise RuntimeError(f"No LinkedIn cookies found in {state_path}")
    return session.cookies


def cookies_from_browser(browser_name):
    if browser_cookie3 is None:
        raise RuntimeError("Install browser-cookie3 first: pip install browser-cookie3")

    loaders = {
        "chrome": browser_cookie3.chrome,
        "edge": browser_cookie3.edge,
        "firefox": browser_cookie3.firefox,
    }
    return loaders[browser_name](domain_name=".linkedin.com")


def fetch_linkedin(url, cookies):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, cookies=cookies, headers=headers, timeout=30)
    response.raise_for_status()
    return response


def parse_html(url, response):
    soup = BeautifulSoup(response.text, "html.parser")

    data = {
        "source_url": url,
        "final_url": response.url,
        "status_code": response.status_code,
        "title": soup.title.get_text(strip=True) if soup.title else None,
        "meta_description": get_meta(soup, "name", "description"),
        "og_title": get_meta(soup, "property", "og:title"),
        "og_description": get_meta(soup, "property", "og:description"),
        "canonical": get_link(soup, "canonical"),
        "text_preview": soup.get_text("\n", strip=True)[:5000],
    }
    return data


def get_meta(soup, attr_name, attr_value):
    tag = soup.find("meta", attrs={attr_name: attr_value})
    return tag.get("content") if tag else None


def get_link(soup, rel):
    tag = soup.find("link", rel=rel)
    return tag.get("href") if tag else None


def save_mongo(args, data):
    if not args.mongo_uri:
        return
    from pymongo import MongoClient

    client = MongoClient(args.mongo_uri)
    collection = client[args.mongo_db][args.mongo_collection]
    collection.update_one({"source_url": data["source_url"]}, {"$set": data}, upsert=True)
    client.close()


def main():
    args = parse_args()

    cookies = cookies_from_playwright_state(args.state)
    cookie_source = args.state

    if cookies is None and args.browser:
        cookies = cookies_from_browser(args.browser)
        cookie_source = args.browser

    if cookies is None:
        raise RuntimeError(
            "No session found. Run `python .\\linkedin_scraper_playwright.py --login` "
            "once, or pass `--browser chrome|edge|firefox`."
        )

    response = fetch_linkedin(args.url, cookies)
    data = parse_html(args.url, response)

    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    save_mongo(args, data)

    print(f"Used session from: {cookie_source}")
    print(f"Saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
