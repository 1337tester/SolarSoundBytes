import pandas as pd
import os

def get_energy_df():

    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'csv', 'IRENA_plus_prediction2024.csv'))

    df = pd.read_csv(
        csv_path)

    data = df[['Unnamed: 0', 'share', 'renewables']]
    data = data.rename(columns={'Unnamed: 0': 'year'})

    return data

data = get_energy_df()
print(data.head())
