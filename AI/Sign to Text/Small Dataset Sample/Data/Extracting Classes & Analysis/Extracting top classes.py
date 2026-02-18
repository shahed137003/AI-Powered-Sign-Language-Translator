import os
import shutil
import re
import pandas as pd
from collections import Counter


def extract_gloss(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Keep part after dash
    if '-' in name:
        name = name.split('-', 1)[1]

    # Remove "seed" if it exists at start
    name = re.sub(r'^seed', '', name, flags=re.IGNORECASE).strip()

    # Remove trailing numbers (e.g., " 1", " 2")
    name = re.sub(r'\s*\d+$', '', name).strip()

    # Convert to uppercase
    return name.upper()


def main():
    # =========================
    # CONFIG
    # =========================
    VIDEO_DIR = r"E:\ASL_Citizen\videos"
    NEW_DIR = r"E:\ASL_Citizen\Top_Classes"
    CSV_PATH = os.path.join(NEW_DIR, "top_classes_counts.csv")

    # Make new folder if it doesn't exist
    os.makedirs(NEW_DIR, exist_ok=True)

    # =========================
    # DEFINE TOP CLASS WORDS
    # =========================
    top_classes = [
        # Family
        "mother", "father", "brother", "sister", "boy", "girl", "grandpa", "aunt", "uncle",
        "baby", "single", "divorce", "grandfather", "grandmother",
        # Places
        "home", "work", "school", "homework", "church",
        # Actions / Movement
        "come", "go", "car", "drive", "in", "out", "about", "up", "off", "down",
        "more", "less", "with", "without", "today", "holiday", "eat", "drink",
        "open", "close", "sit", "stand", "understand", "run", "walk", "sleep",
        "want", "need", "see", "hear", "play", "wait",
        # Time
        "day", "night", "light", "dark", "week", "month", "year",
        "will", "before", "after", "finish", "now", "yesterday",
        # Temperature
        "hot", "cold",
        # Food / Drink / Utensils
        "pizza", "milk", "hamburger", "hotdog", "egg", "apple", "cheese", "drink",
        "spoon", "fork", "cup", "cereal", "water", "candy", "cookie", "hungry",
        # Clothing / Hygiene
        "shirt", "pants", "socks", "shoes", "underwear", "wash", "hurt",
        "bathroom", "toothbrush", "brush", "sleep", "nice", "clean",
        # Feelings / Emotions
        "happy", "angry", "sad", "sorry", "cry", "like", "good", "bad", "love",
        # Requests / Questions
        "please", "excuse", "help", "who", "what", "when", "where", "why", "how", "stop",
        # Amounts / Size
        "big", "tall", "full", "more",
        # Colors
        "blue", "green", "yellow", "red", "brown", "orange", "gold", "silver",
        # Money
        "dollars", "cost",
        # Animals
        "cat", "dog", "bird", "horse", "cow", "sheep", "pig", "bug",
        # Pronouns
        "they", "you", "your",
        # Basic Modifiers
        "yes", "no", "cannot", "can", "not",
        # Other / Misc
        "here", "there", "child", "welcome", "same", "friend", "teacher"
    ]

    # Uppercase for matching
    top_classes_upper = [w.upper() for w in top_classes]

    # =========================
    # FILTER, COPY, RENAME
    # =========================
    gloss_counter = Counter()

    for file in os.listdir(VIDEO_DIR):
        if file.lower().endswith(".mp4"):
            gloss = extract_gloss(file)

            if gloss in top_classes_upper:
                gloss_counter[gloss] += 1
                count = gloss_counter[gloss]

                new_name = gloss if count == 1 else f"{gloss} {count}"
                new_file_path = os.path.join(NEW_DIR, new_name + ".mp4")

                shutil.copy(os.path.join(VIDEO_DIR, file), new_file_path)

    # =========================
    # SAVE CSV
    # =========================
    df_top = pd.DataFrame(list(gloss_counter.items()), columns=["GLOSS", "COUNT"])
    df_top = df_top.sort_values(by="COUNT", ascending=False).reset_index(drop=True)
    df_top.to_csv(CSV_PATH, index=False)

    print(f"Copied {sum(gloss_counter.values())} videos to {NEW_DIR}")
    print(f"CSV saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
