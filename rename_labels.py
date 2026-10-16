import pandas as pd
import shutil
from pathlib import Path

# Existing dataset
dataset = Path("hand_sign_data.csv")

# Create a backup before making changes
backup = Path("hand_sign_data_backup.csv")

if not dataset.exists():
    raise FileNotFoundError("hand_sign_data.csv was not found.")

if not backup.exists():
    shutil.copy2(dataset, backup)
    print("Backup created:", backup)

data = pd.read_csv(dataset)

print("\nCurrent labels:")
print(data["label"].value_counts().to_string())

# EDIT THIS DICTIONARY ONLY AFTER CHECKING YOUR GESTURES.
# These are examples, not automatic corrections.
rename_labels = {
    "I LOVE YOU❤️": "I LOVE YOU",
    "rock_on": "LETS ROCK BUDDY!",
    "I HATE YOU BRO💔": "I HATE YOU BRO",
    "I PURPLE YOU💜": "I PURPLE YOU",
    "WELL DONE👍": "WELL DONE",
    "HI! HOW ARE YOU✋": "HI! HOW ARE YOU",
    "PLEASE CALL ME🤙": "PLEASE CALL ME",
    "TOGETHER WE ARE STRONG✊": "TOGETHER WE ARE STRONG",
    "YES! WE DID IT✌️": "YES! WE DID IT",
    "ALL IS WELL👌": "ALL IS WELL",
    "NOT GOOD👎": "NOT GOOD",
}

if not rename_labels:
    print("\nNo labels renamed yet.")
    print("Edit rename_labels after confirming the correct gestures.")
else:
    print("\nPlanned changes:")
    for old, new in rename_labels.items():
        print(f"{old} -> {new}")

    confirm = input("\nType YES to apply these changes: ")

    if confirm == "YES":
        data["label"] = data["label"].replace(rename_labels)
        data.to_csv(dataset, index=False)
        print("Labels updated successfully.")
    else:
        print("Cancelled. Dataset was not changed.")
