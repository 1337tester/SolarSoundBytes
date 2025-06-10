import pandas as pd
import os


def create_df_of_twitter_result():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'data_test', 'twitter_sentiment_analysis_UTF8.csv')

    data = pd.read_csv(file_path,  encoding='utf-8')

    df = data[['createdAt', 'sentiment', 'confidence score']]
    df = df.rename(columns={'createdAt': 'published'})
    return df


# test = create_df_of_twitter_result()
# print(test.head())
