import csv
import os

def load_usernames(file_path):
    usernames = set()

    with open(file_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Try to find "username" column, else use first column
        try:
            username_index = header.index("username")
        except ValueError:
            username_index = 0

        for row in reader:
            if row and len(row) > username_index:
                usernames.add(row[username_index].strip().lower())

    return usernames


# ---------- USER INPUT ----------
following_file = input("Enter following CSV file path: ").strip().strip('"').strip("'")
followers_file = input("Enter followers CSV file path: ").strip().strip('"').strip("'")


# ---------- VALIDATION ----------
if not os.path.isfile(following_file):
    raise FileNotFoundError(f"Following file not found: {following_file}")

if not os.path.isfile(followers_file):
    raise FileNotFoundError(f"Followers file not found: {followers_file}")


# ---------- LOAD DATA ----------
following = load_usernames(following_file)
followers = load_usernames(followers_file)

all_users = sorted(following | followers)


# ---------- PREPARE OUTPUT ----------
rows = []

for user in all_users:
    i_follow = "Yes" if user in following else "No"
    follows_me = "Yes" if user in followers else "No"

    if i_follow == "Yes" and follows_me == "Yes":
        status = "Mutual"
    elif i_follow == "Yes" and follows_me == "No":
        status = "Not Following You"
    elif i_follow == "No" and follows_me == "Yes":
        status = "Follows You"
    else:
        status = "No Relationship"

    rows.append([user, i_follow, follows_me, status])


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
