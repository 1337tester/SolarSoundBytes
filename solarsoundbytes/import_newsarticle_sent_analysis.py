import pandas as pd
import os


def create_df_of_newsarticle_result():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'data_test', 'cleantech_predictions_with_confidence _UTF8.xlsx')

    data = pd.read_excel(file_path)

    df = data[['Date Published', 'predicted_sentiment', 'confidence_score']]
    df = df.rename(columns={'Date Published': 'published',
                            'predicted_sentiment': 'sentiment',
                            'confidence_score': 'confidence score'})
    return df


# test = create_df_of_newsarticle_result()
# print(test.head())
