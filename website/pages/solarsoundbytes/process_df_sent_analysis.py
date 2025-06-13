import pandas as pd


from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result
from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result


df_twitter = create_df_of_twitter_result()
df_news = create_df_of_newsarticle_result()

def count_sent_per_quarter(df):
    df['sentiment'] = df['sentiment'].str.lower()
    df['published'] = pd.to_datetime(df['published'])
    df['quarter'] = df['published'].dt.to_period('Q')
    df['three_month_block'] = (df['published'].dt.year * 100 +
                                    ((df['published'].dt.month - 1) // 3) * 3 + 1)
    df_counts = df.groupby(['quarter', 'sentiment']).size().reset_index(name='count')
    df_counts['quarter_start'] = df_counts['quarter'].dt.start_time
    return df_counts


def agg_monthly_sent_analysis(df):
    df['published'] = pd.to_datetime(df['published'])
    df['sentiment'] = df['sentiment'].str.lower()
    df['date'] = df['published'].dt.to_period('M').dt.to_timestamp()
    #   Gruppieren nach Monat und Sentiment, dann aggregieren
    agg_df = df.groupby(['date', 'sentiment']).agg(
        count=('sentiment', 'size'),
        confidence_score_mean=('confidence score', 'mean')
    ).reset_index()
    return agg_df

# df_counts_twitter = count_sent_per_quarter(df_twitter)
# df_counts_news = count_sent_per_quarter(df_news)
# agg_df_twitter = agg_monthly_sent_analysis(df_twitter)
# agg_df_news = agg_monthly_sent_analysis(df_news)
# print(agg_df_twitter.head())
# print(agg_df_news.head())
