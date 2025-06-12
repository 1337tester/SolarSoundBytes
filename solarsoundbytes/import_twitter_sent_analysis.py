import pandas as pd
import os


def create_df_of_twitter_result():
    # base_path = os.path.dirname(__file__)
    # file_path = os.path.join(base_path, 'data_test', 'twitter_sentiment_analysis_UTF8.csv')
    # file_path = '../data/csv/twitter_sentiment_analysis_UTF8.csv'
    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data_input_streamlit', 'twitter_sentiment_analysis_UTF8.csv'))

    data = pd.read_csv(csv_path,  encoding='utf-8')

    df = data[['createdAt', 'sentiment', 'confidence score']]
    df = df.rename(columns={'createdAt': 'published'})
    return df


test = create_df_of_twitter_result()
