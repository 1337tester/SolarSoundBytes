import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from gtts import gTTS

from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result_events
from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result

from solarsoundbytes.text_creation.create_text import create_text_from_sent_analy_df

st.set_page_config(layout="wide")
st.subheader('Solar production > 1 TW')
# --- load api key from streamlit secrets .streamlit ---
try:
    api_key_from_secrets = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Error: OpenAI API Key not found in .streamlit/secrets.toml")
    st.info("Please set your OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
    st.stop() # Stoppt die App, wenn der Schlüssel fehlt


start_date = pd.to_datetime("2022-12-30")
end_date = pd.to_datetime("2023-01-01")



# --- DATA SOURCE ---
df_twitter = create_df_of_twitter_result_events()
df_news = create_df_of_newsarticle_result()


# Filter datasets

# Initialize the figure
fig = go.Figure()

# --- News Sentiment Bubble Chart Trace ---
df_news['date'] = pd.to_datetime(df_news['date'])
df_news_filtered = df_news[(df_news['date'] >= start_date) & (df_news['date'] <= end_date)].copy()

# Konvertiere Datum und extrahiere Monat
df_news_filtered['date'] = pd.to_datetime(df_news_filtered['date'])
#df_news_filtered['month'] = df_news_filtered['date'].dt.to_period('M').dt.to_timestamp()

# Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
df_news_filtered['correct_prob'] = df_news_filtered[['pos_score', 'neg_score']].max(axis=1)

# Aggregiere nach Monat
monthly_stats_news = df_news_filtered.groupby('date').agg(
    mean_correct_prob=('correct_prob', 'mean'),
    mean_pos_score=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_correct_prob=('correct_prob', 'std'),
).reset_index()

# Create scatter trace for news sentiment
# Handle potential NaN in std_correct_prob for months with only one data point
# If a month has only one news item, std dev is NaN. Set to 0 for error bar.
monthly_stats_news['std_correct_prob'] = monthly_stats_news['std_correct_prob'].fillna(0)

custom_r_g_b_colorscale = [
    [0.0, 'rgb(255,0,0)'],   # red
    [0.25, 'rgb(255,165,0)'], # Orange
    [0.5, 'rgb(75,0,130)'],   # indigo
    [0.75, 'rgb(0,0,255)'],  # Blue
    [1.0, 'rgb(0,128,0)']   # green
]


# --- News Sentiment Bubble Chart Trace ---
fig.add_trace(go.Scatter(
    x=monthly_stats_news['date'],
    y=monthly_stats_news['std_correct_prob'],
    mode='markers',
    marker=dict(
        size=monthly_stats_news['count'], # Using 'count' for size
        sizemode='area',
        sizeref=2. * monthly_stats_news['count'].max() / (40. ** 2),
        sizemin=4,
        color=monthly_stats_news['mean_pos_score'], # Color by mean_pos_score
        colorscale=custom_r_g_b_colorscale,
        cmin=0,
        cmax=1,
        showscale=True,
    ),
    name='News Sentiment',
    # visible='legendonly',
    yaxis='y1', # Left y-axis
    # Customdata for hover information (excluding std_correct_prob since it's now an error bar)
    customdata=monthly_stats_news[['std_correct_prob', 'mean_pos_score']].values,
    hovertemplate=(
        "<b>Day:</b> %{x|%Y-%m-%d}<br>" +
        "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
        "<b>Mean Pos-score:</b> %{customdata[1]:.2f}<br>" +
        "<b>News Count:</b> %{marker.size}<extra></extra>"
    )
))
df_twitter['date'] = pd.to_datetime(df_twitter['date']).dt.tz_convert(None)
df_twitter_filtered = df_twitter[(df_twitter['date'] >= start_date) & (df_twitter['date'] <= end_date)].copy()

# Konvertiere Datum und extrahiere Hour
df_twitter_filtered['date'] = pd.to_datetime(df_twitter_filtered['date'])
df_twitter_filtered['hour'] = df_twitter_filtered['date'].dt.to_period('h').dt.to_timestamp()
# Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
df_twitter_filtered['correct_prob'] = df_twitter_filtered[['pos_score', 'neg_score']].max(axis=1)

#
hourly_stats_twitter = df_twitter_filtered.groupby('hour').agg(
    mean_correct_prob=('correct_prob', 'mean'),
    mean_pos_score=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_correct_prob=('correct_prob', 'std'),
).reset_index()

# --- Twitter Sentiment Bubble Chart Trace (main trace) ---
fig.add_trace(go.Scatter(
    x=hourly_stats_twitter['hour'],
    y=hourly_stats_twitter['std_correct_prob'], # Use the y-values with random offset
    mode='markers',
    marker=dict(
        symbol='diamond',  # Change marker to squares
        size=hourly_stats_twitter['count'], # Using 'count' for size
        sizemode='area',
        sizeref=2. * hourly_stats_twitter['count'].max() / (40. ** 2),
        sizemin=4,
        color=hourly_stats_twitter['mean_pos_score'], # Color by mean_pos_score
        colorscale=custom_r_g_b_colorscale,
        cmin=0, # Keep original cmin
        cmax=1,  # Keep original cmax
        showscale=True,

    ),
    # visible='legendonly',
    name='Twitter Sentiment',
    yaxis='y1', # Left y-axis
    customdata=hourly_stats_twitter[['mean_pos_score', 'std_correct_prob']].values, # Include std_correct_prob in customdata
    hovertemplate=(
        "<b>Hour:</b> %{x|%Y-%m-%d %H:00}<br>" +
        "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
        "<b>Mean Pos-score:</b> %{customdata[0]:.2f}<br>" +
        "<b>Tweets Count:</b> %{marker.size}<extra></extra>"
    )
))


fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis1=dict(
        title='',
        side='left', # News Sentiment on the left
        showgrid=True,
        anchor='free', # Allow free positioning
        overlaying='y', # Overlay on the primary y-axis
        autorange='reversed',
        position=0 # Position of the left y-axis (0 is far left)
    ),
    legend=dict(x=0.01, y=0.99), # Legend position
    height=600,
    margin=dict(r=200),
    # Make sure to set proper ranges if necessary or Plotly will auto-scale
    # yaxis_range=[min_val_sp500, max_val_sp500],
    # yaxis2_range=[min_val_renewables, max_val_renewables],
    # yaxis3_range=[min_val_sentiment, max_val_sentiment],
)



st.plotly_chart(fig, use_container_width=True)
