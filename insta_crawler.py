import time
import random
from pymongo import MongoClient
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.instagram.com"
TARGET_USER = "instagram"

MONGO_URI = "mongodb://localhost:27017"

client = MongoClient(MONGO_URI)
db = client["instagram"]

profiles_col = db["profiles"]
posts_col = db["posts"]
comments_col = db["comments"]
followers_col = db["followers"]
following_col = db["following"]


def random_sleep(a=2, b=5):
    time.sleep(random.uniform(a, b))


def auto_scroll(page, loops=10):
    for _ in range(loops):
        page.mouse.wheel(0, 8000)
        random_sleep(2, 4)


def scrape_profile(page, username):

    page.goto(f"{BASE_URL}/{username}/")
    random_sleep(4, 6)

    profile = {"username": username}

    try:
        stats = page.locator("header li span")

        profile["posts"] = stats.nth(0).inner_text()
        profile["followers"] = stats.nth(1).inner_text()
        profile["following"] = stats.nth(2).inner_text()

        profile["bio"] = page.locator("header section").inner_text()

    except:
        pass

    profiles_col.update_one(
        {"username": username},
        {"$set": profile},
        upsert=True
    )


def collect_post_links(page):

    links = set()

    anchors = page.locator("article a")

    for i in range(anchors.count()):
        href = anchors.nth(i).get_attribute("href")
        if href and "/p/" in href:
            links.add(BASE_URL + href)

    return list(links)


def scrape_comments(page, post_url):

    try:
        page.locator("svg[aria-label='Load more comments']").click()
    except:
        pass

    comments = page.locator("ul ul")

    docs = []

    for i in range(comments.count()):
        try:
            user = comments.nth(i).locator("a").first.inner_text()
            text = comments.nth(i).locator("span").inner_text()

            docs.append({
                "post_url": post_url,
                "user": user,
                "comment": text
            })

        except:
            pass

    if docs:
        comments_col.insert_many(docs)


def scrape_post(page, url, username):

    page.goto(url)
    random_sleep(3, 5)

    post = {
        "username": username,
        "url": url
    }

    try:
        post["caption"] = page.locator("h1").inner_text()
    except:
        pass

    try:
        post["likes"] = page.locator("section span").first.inner_text()
    except:
        pass

    try:
        post["date"] = page.locator("time").get_attribute("datetime")
    except:
        pass

    posts_col.update_one(
        {"url": url},
        {"$set": post},
        upsert=True
    )

    scrape_comments(page, url)


def scrape_posts(page, username):

    page.goto(f"{BASE_URL}/{username}/")
    random_sleep(4, 6)

    auto_scroll(page, 8)

    links = collect_post_links(page)

    for link in links[:20]:

        try:
            scrape_post(page, link, username)
            random_sleep(2, 4)

        except:
            pass


def scroll_dialog(page):

    dialog = page.locator("div[role='dialog']")

    for _ in range(20):
        dialog.evaluate("el => el.scrollTop = el.scrollHeight")
        random_sleep(1, 2)


def scrape_follow(page, username, mode="followers"):

    page.goto(f"{BASE_URL}/{username}/{mode}/")
    random_sleep(5, 7)

    scroll_dialog(page)

    users = page.locator("div[role='dialog'] a")

    docs = []

    for i in range(users.count()):

        try:
            name = users.nth(i).inner_text()

            docs.append({
                "username": username,
                mode: name
            })

        except:
            pass

    if docs:
        if mode == "followers":
            followers_col.insert_many(docs)
        else:
            following_col.insert_many(docs)


def login(page, username, password):

    page.goto(f"{BASE_URL}/accounts/login/")
    random_sleep(5, 6)

    page.fill("input[name='username']", username)
    page.fill("input[name='password']", password)

    page.click("button[type='submit']")

    random_sleep(8, 10)


def main():

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context()

        page = context.new_page()

        # Optional login
        # login(page, "your_username", "your_password")

        scrape_profile(page, TARGET_USER)

        scrape_posts(page, TARGET_USER)

        scrape_follow(page, TARGET_USER, "followers")

        scrape_follow(page, TARGET_USER, "following")

        browser.close()


if __name__ == "__main__":
    main()