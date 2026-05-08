import csv
import os

USERNAME_COLUMNS = ("username", "user_name", "userName", "UserName")
ID_COLUMNS = ("id", "user_id", "userId", "pk")


def normalize_header(value):
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def find_column(header, candidates):
    normalized_header = [normalize_header(column) for column in header]
    normalized_candidates = {normalize_header(candidate) for candidate in candidates}

    for index, column in enumerate(normalized_header):
        if column in normalized_candidates:
            return index

    return None


def load_users(file_path):
    users = {}

    with open(file_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return users

        username_index = find_column(header, USERNAME_COLUMNS)
        id_index = find_column(header, ID_COLUMNS)

        if username_index is None and id_index is None:
            raise ValueError(
                f"No username/userName or id column found in {file_path}. "
                f"Available columns: {', '.join(header)}"
            )

        for row in reader:
            if not row:
                continue

            username = ""
            user_id = ""

            if username_index is not None and len(row) > username_index:
                username = row[username_index].strip()

            if id_index is not None and len(row) > id_index:
                user_id = row[id_index].strip()

            if not username and not user_id:
                continue

            # Match by stable Instagram id when available, but display username.
            key = (user_id or username).lower()
            display_name = username or user_id
            users[key] = display_name

    return users


# ---------- USER INPUT ----------
following_file = input("Enter following CSV file path: ").strip().strip('"').strip("'")
followers_file = input("Enter followers CSV file path: ").strip().strip('"').strip("'")


# ---------- VALIDATION ----------
if not os.path.isfile(following_file):
    raise FileNotFoundError(f"Following file not found: {following_file}")

if not os.path.isfile(followers_file):
    raise FileNotFoundError(f"Followers file not found: {followers_file}")


# ---------- LOAD DATA ----------
following = load_users(following_file)
followers = load_users(followers_file)

all_users = sorted(set(following) | set(followers))


# ---------- PREPARE OUTPUT ----------
rows = []

for user in all_users:
    i_follow = "Yes" if user in following else "No"
    follows_me = "Yes" if user in followers else "No"
    username = following.get(user) or followers.get(user)

    if i_follow == "Yes" and follows_me == "Yes":
        status = "Mutual"
    elif i_follow == "Yes" and follows_me == "No":
        status = "Not Following You"
    elif i_follow == "No" and follows_me == "Yes":
        status = "Follows You"
    else:
        status = "No Relationship"

    rows.append([username, i_follow, follows_me, status])


# ---------- TERMINAL TABLE ----------
USERNAME_WIDTH = 30

print()
print(f"{'Username':<{USERNAME_WIDTH}}  {'I_Follow':<10} {'Follows_Me':<12} Status")
print("-" * (USERNAME_WIDTH + 36))

for r in rows:
    print(f"{r[0]:<{USERNAME_WIDTH}}  {r[1]:<10} {r[2]:<12} {r[3]}")


# ---------- SAVE TO EXCEL (CSV) ----------
output_file = "instagram_follow_report.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Username", "I_Follow", "Follows_Me", "Status"])
    writer.writerows(rows)

print(f"\n✅ Excel file created: {output_file}")
