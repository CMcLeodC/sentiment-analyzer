import os
import random
import re
import pandas as pd

# Load and clean reviews from folder
def load_reviews(folder_path, label, max_count=1000):
    reviews = []
    folder = os.path.join(folder_path, label)
    files = os.listdir(folder)
    random.shuffle(files)

    for file in files[:max_count]:
        with open(os.path.join(folder, file), encoding='utf-8') as f:
            text = f.read()
            text = clean_text(text)
            reviews.append((text, label))
    return reviews

# Basic cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-letter characters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load 1000 pos + 1000 neg
pos_reviews = load_reviews("aclImdb/train", "pos", max_count=1000)
neg_reviews = load_reviews("aclImdb/train", "neg", max_count=1000)

# Combine and shuffle
all_reviews = pos_reviews + neg_reviews
random.shuffle(all_reviews)

# Save as DataFrame
df = pd.DataFrame(all_reviews, columns=["review", "label"])
df.to_csv("imdb_small.csv", index=False)

print(f"Saved {len(df)} reviews to imdb_small.csv ✅")
