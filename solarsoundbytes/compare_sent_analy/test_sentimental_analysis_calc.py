import pandas as pd
import numpy as np
from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result
from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result


def sent_analy_to_values_twitter():
    # labels = np.random.choice(['pos', 'neg', 'neu'], size=n)

    # values = np.random.uniform(low=0.75, high=0.99, size=n)

    # start_date = pd.to_datetime('2022-01-01')
    # end_date = pd.to_datetime('2024-12-31')

    # # Liste zufälliger Datumswerte (z. B. aus 100 möglichen Tagen, mehrfach erlaubt)
    # date_choices = pd.date_range(start=start_date, end=end_date, periods=100)
    # dates = np.random.choice(date_choices, size=n)

    # df = pd.DataFrame({
    #     'label': labels,
    #     'values': values,
    #     'date': dates,

    # })

    df = create_df_of_twitter_result()
        ## use the dataframe from Fadri with create_df_of_twitter_result()
        ## instead of the fake one!!!!
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['month_year'] = df['createdAt'].dt.strftime('%Y-%m')
    df['quarter'] = df['createdAt'].dt.to_period('Q').astype(str)
    ## convert result of sent analysis into numbers

    df['sentiment_value'] = df['sentiment'].map({
        'neutral': 0,
        'negative': -1,
        'positive': 1,
    })

    # multiply the numbers with the confident interval
    df['multiplication'] = df['sentiment_value'] * df['confidence score']

    df['positive'] = df[df['multiplication'] > 0]['multiplication']
    df['negative'] = df[df['multiplication'] < 0]['multiplication']

    # mean  positiv/negativ monthly
    def monthly_mean_pos_neg(gruppe):
        positive = gruppe[gruppe['multiplication'] > 0]['multiplication'].mean()
        negative = gruppe[gruppe['multiplication'] < 0]['multiplication'].mean()
        return pd.Series({
            'mean_pos': positive,
            'mean_neg': negative,
        })

    # Gruppieren nach Woche + anwenden
    months_all_df = df.groupby('month_year').apply(monthly_mean_pos_neg).reset_index()

    ##group the results by months and calculate the mean
    months_pos_neg_df = df.groupby('month_year')['multiplication'].mean().reset_index()

    months_df = pd.merge(months_all_df, months_pos_neg_df, on='month_year', how='inner')
    # quarters_df = pd.merge()
    return months_df


def sent_analy_to_values_newsarticle():

    df = create_df_of_newsarticle_result()
        ## use the dataframe from Fadri with create_df_of_twitter_result()
        ## instead of the fake one!!!!
    df['Date Published'] = pd.to_datetime(df['Date Published'])
    df['month_year'] = df['Date Published'].dt.strftime('%Y-%m')
    df['quarter'] = df['Date Published'].dt.to_period('Q').astype(str)
    ## convert result of sent analysis into numbers

    df['sentiment_value'] = df['predicted_sentiment'].map({
        'neutral': 0,
        'negative': -1,
        'positive': 1,
    })

    # multiply the numbers with the confident interval
    df['multiplication'] = df['sentiment_value'] * df['confidence_score']

    df['positive'] = df[df['multiplication'] > 0]['multiplication']
    df['negative'] = df[df['multiplication'] < 0]['multiplication']

    # mean  positiv/negativ monthly
    def monthly_mean_pos_neg(gruppe):
        positive = gruppe[gruppe['multiplication'] > 0]['multiplication'].mean()
        negative = gruppe[gruppe['multiplication'] < 0]['multiplication'].mean()
        return pd.Series({
            'mean_pos': positive,
            'mean_neg': negative,
        })

    # Gruppieren nach Woche + anwenden
    months_all_df = df.groupby('month_year').apply(monthly_mean_pos_neg).reset_index()

    ##group the results by months and calculate the mean
    months_pos_neg_df = df.groupby('month_year')['multiplication'].mean().reset_index()

    months_df = pd.merge(months_all_df, months_pos_neg_df, on='month_year', how='inner')
    # quarters_df = pd.merge()
    months_df = months_df.replace([np.nan, np.inf, -np.inf], None)
    months_df.index = months_df.index.astype(str)

    return months_df

def create_output_interface():
    result_twitter = sent_analy_to_values_twitter()
    result_news = sent_analy_to_values_newsarticle()
    return result_twitter, result_news

test = create_output_interface()
