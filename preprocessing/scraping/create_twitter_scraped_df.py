import os
import json
import pandas as pd
from datetime import datetime
from glob import glob

# Helper function to extract fields from each tweet dictionary
def extract_tweet_data(tweet, reference_date):
    def safe_get(dct, key, default=None):
        return dct.get(key, default)

    def get_user_mentions(entities):
        mentions = safe_get(entities, "user_mentions", [])
        id_strs = "~~".join([m.get("id_str", "") for m in mentions])
        indices_0 = mentions[0]["indices"][0] if len(mentions) > 0 else None
        indices_1 = mentions[1]["indices"][1] if len(mentions) > 1 else None
        name = "~~".join([m.get("name", "") for m in mentions])
        screen_name = "~~".join([m.get("screen_name", "") for m in mentions])
        return id_strs, indices_0, indices_1, name, screen_name

    entities = tweet.get("entities", {})
    user_mentions = get_user_mentions(entities)

    return {
        "id": safe_get(tweet, "id"),
        "url": safe_get(tweet, "url"),
        "text": safe_get(tweet, "text"),
        "source": safe_get(tweet, "source"),
        "date": reference_date,
        "time of day": safe_get(tweet, "createdAt").split(" ")[4] if tweet.get("createdAt") else None,
        "location": tweet.get("location", None),
        "retweetCount": safe_get(tweet, "retweetCount"),
        "replyCount": safe_get(tweet, "replyCount"),
        "likeCount": safe_get(tweet, "likeCount"),
        "quoteCount": safe_get(tweet, "quoteCount"),
        "viewCount": safe_get(tweet, "viewCount"),
        "createdAt": safe_get(tweet, "createdAt"),
        "bookmarkCount": safe_get(tweet, "bookmarkCount"),
        "isReply": safe_get(tweet, "isReply"),
        "inReplyToId": safe_get(tweet, "inReplyToId"),
        "conversationId": safe_get(tweet, "conversationId"),
        "inReplyToUserId": safe_get(tweet, "inReplyToUserId"),
        "inReplyToUsername": safe_get(tweet, "inReplyToUsername"),
        "isPinned": safe_get(tweet, "isPinned"),
        "user_mentions_id_str": user_mentions[0],
        "user_mentions_indices_0": user_mentions[1],
        "user_mentions_indices_1": user_mentions[2],
        "user_mentions_name": user_mentions[3],
        "user_mentions_screen_name": user_mentions[4],
        "reply_to_user_results": safe_get(tweet, "reply_to_user_results"),
        "quoted_tweet_results": safe_get(tweet, "quoted_tweet_results"),
        "quoted_tweet": safe_get(tweet, "quoted_tweet"),
        "retweeted_tweet": safe_get(tweet, "retweeted_tweet"),
        "isConversationControlled": safe_get(tweet, "isConversationControlled"),
        "searchTermIndex": safe_get(tweet, "searchTermIndex")
    }

def convert_twitter_datetime(date_str):
    try:
        return pd.to_datetime(date_str, format='%a %b %d %H:%M:%S %z %Y')
    except (ValueError, TypeError):
        return pd.NaT

# # select only 2 files manually to test output.csv
# json_files = [
#     "../../data/json/dataset_2022-01-02_2025-06-03_22-22-56-837.json",
#     "../../data/json/dataset_2022-02-02_2025-06-03_22-57-32-967.json"
# ]

# process all files
json_files = sorted(glob("../../data/json/*.json"))

all_data = []

for filepath in json_files:
    with open(filepath, "r", encoding="utf-8") as f:
        tweets = json.load(f)
        date_part = os.path.basename(filepath).split("_")[1]  # e.g. '2022-01-02'
        for tweet in tweets:
            extracted = extract_tweet_data(tweet, date_part)
            all_data.append(extracted)

twitter_scraped_df = pd.DataFrame(all_data)

# Convert createdAt to datetime
twitter_scraped_df['createdAt'] = twitter_scraped_df['createdAt'].apply(convert_twitter_datetime)

# Sort by datetime index
twitter_scraped_df.sort_values('createdAt', inplace=True)

# Ensure output folder exists and create output path
output_folder = "../../data/csv"
os.makedirs(output_folder, exist_ok=True)
csv_output_path = os.path.join(output_folder, "twitter_scraped_df.csv")

# Set index and save CSV
twitter_scraped_df.set_index("createdAt", inplace=True)  # use datetime as index
twitter_scraped_df.to_csv(csv_output_path)

print(f"✅ CSV file saved to: {csv_output_path}")
