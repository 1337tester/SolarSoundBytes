import pandas as pd
import numpy as np


np.random.seed(42)  # für reproduzierbare Ergebnisse
n = 1000  # Anzahl der Zeilen


labels = np.random.choice(['pos', 'neg', 'neu'], size=n)

values = np.random.uniform(low=0.75, high=0.99, size=n)

start_date = pd.to_datetime('2024-12-01')
end_date = pd.to_datetime('2024-12-31')

# Liste zufälliger Datumswerte (z. B. aus 100 möglichen Tagen, mehrfach erlaubt)
date_choices = pd.date_range(start=start_date, end=end_date, periods=100)
dates = np.random.choice(date_choices, size=n)

df = pd.DataFrame({
    'label': labels,
    'values': values,
    'date': dates,

})

df['week_year'] = df['date'].dt.strftime('%Y-%U')  # %U = Woche (Sonntag als Start)


df['label_value'] = df['label'].map({
    'neu': 0,
    'neg': -1,
    'pos': 1,
})

df['multiplication'] = df['label_value'] * df['values']


wochen_df = df.groupby('week_year')['multiplication'].mean().reset_index()
wochen_df.head()
