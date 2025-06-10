import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px

from streamlit_plotly_events import plotly_events


from gtts import gTTS

from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result
from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result

# from solarsoundbytes.compare_sent_analy.test_sentimental_analysis_calc import create_output_interface
# from solarsoundbytes.text_creation.create_text import create_text_from_sent_analy_df
from solarsoundbytes.process_sp500_df import preprocess_sp500_df
from solarsoundbytes.process_df_sent_analysis import count_sent_per_quarter
from solarsoundbytes.process_df_sent_analysis import agg_monthly_sent_analysis

st.set_page_config(layout="wide")

# --- load api key from streamlit secrets .streamlit ---
try:
    api_key_from_secrets = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Error: OpenAI API Key not found in .streamlit/secrets.toml")
    st.info("Please set your OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
    st.stop() # Stoppt die App, wenn der Schlüssel fehlt


# --- DATA SOURCE ---
df_twitter = create_df_of_twitter_result()
df_news = create_df_of_newsarticle_result()
counts_news = count_sent_per_quarter(df_news)
counts_twitter = count_sent_per_quarter(df_twitter)
monthly_sp500 = preprocess_sp500_df()

# Quarterly GDP values
gdp = {
    '2022': 5200,
    '2023': 5500,
    '2024': 5750,
}


# Prepare GDP for quarterly plotting (flat line per quarter)
df_gdp_yearly = pd.DataFrame({'year': list(gdp.keys()), 'gdp': list(gdp.values())})
df_gdp_yearly['year'] = pd.to_numeric(df_gdp_yearly['year']) # Ensure year is numeric for comparisons

# Create one row per quarter with the same GDP value (although not used directly in current plot)
quarters_gdp = ['Q1', 'Q2', 'Q3', 'Q4']
df_gdp = pd.DataFrame([
    {'quarter': f"{year}-Q{q}", 'gdp': value}
    for year, value in gdp.items()
    for q in range(1, 5)
])
df_gdp['quarter_start'] = df_gdp['quarter'].apply(lambda x: pd.to_datetime(f"{x[:4]}-{(int(x[-1])-1)*3 + 1:02d}-01"))


# Generate list of quarters (e.g. "2022 Q1", ..., "2024 Q4")
def generate_quarters(start_year, end_year):
    return [f"{year} Q{q}" for year in range(start_year, end_year + 1) for q in range(1, 5)]

quarters_list = generate_quarters(2022, 2024)

# User selection for quarter range
col1, col2 = st.columns(2)
with col1:
    selected_start = st.selectbox("Start Quarter", quarters_list, index=0)
with col2:
    selected_end = st.selectbox("End Quarter", quarters_list, index=len(quarters_list) - 1)

# Convert quarter strings to real date ranges
def quarter_to_dates(q_str):
    year, q = map(int, q_str.split(" Q"))
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    start_date = pd.to_datetime(f"{year}-{start_month:02d}-01")
    end_date = pd.to_datetime(f"{year}-{end_month:02d}-01") + pd.offsets.MonthEnd(1)
    return start_date, end_date

start_date, _ = quarter_to_dates(selected_start)
_, end_date = quarter_to_dates(selected_end)

# Ensure start < end
if start_date > end_date:
    st.error("Start quarter must be before end quarter.")
    st.stop()

# Filter datasets
filtered_sp500 = monthly_sp500[(monthly_sp500['month'] >= start_date) & (monthly_sp500['month'] <= end_date)]
filtered_counts_twitter = counts_twitter[(counts_twitter['quarter_start'] >= start_date) & (counts_twitter['quarter_start'] <= end_date)]
filtered_counts_news = counts_news[(counts_news['quarter_start'] >= start_date) & (counts_news['quarter_start'] <= end_date)]

# --- Plot- ---
fig = go.Figure()

# S&P 500 line
fig.add_trace(go.Scatter(
    x=filtered_sp500['month'], y=filtered_sp500['Price'],
    name='S&P 500',
    yaxis='y1',
    mode='lines+markers',
    line=dict(color='blue')
))

# GDP line
first_gdp = True

for _, row in df_gdp_yearly.iterrows():
    year = row['year']
    value = row['gdp']
    year_start = pd.to_datetime(f"{year}-01-01")
    year_end = pd.to_datetime(f"{year}-12-31")

    line_start = max(start_date, year_start)
    line_end = min(end_date, year_end)

    if line_start < line_end:
        fig.add_trace(go.Scatter(
            x=[line_start, line_end],
            y=[value, value],
            name='GDP',
            yaxis='y2',
            mode='lines',
            line=dict(color='orange', width=4),
            showlegend=first_gdp
        ))
        first_gdp = False



agg_counts_bubbles_twitter = filtered_counts_twitter.pivot_table(
    index='quarter_start',
    columns='sentiment',
    values='count',
    fill_value=0
).reset_index()

agg_counts_bubbles_news = filtered_counts_news.pivot_table(
    index='quarter_start',
    columns='sentiment',
    values='count',
    fill_value=0
).reset_index()


agg_counts_bubbles_twitter['total_count'] = agg_counts_bubbles_twitter['positive'] + agg_counts_bubbles_twitter['neutral'] + agg_counts_bubbles_twitter['negative']
agg_counts_bubbles_twitter['negative_neutral_count'] = agg_counts_bubbles_twitter['neutral'] + agg_counts_bubbles_twitter['negative']
agg_counts_bubbles_news['total_count'] = agg_counts_bubbles_news['positive']    # + agg_counts_bubbles_news['neutral'] + agg_counts_bubbles_news['negative']
agg_counts_bubbles_news['negative_neutral_count'] = 0   #agg_counts_bubbles_news['neutral'] + agg_counts_bubbles_news['negative']


bubble_y_position_twitter = filtered_sp500['Price'].min() - 2000 if not filtered_sp500.empty else 1000
bubble_y_position_news = filtered_sp500['Price'].min() - 1500 if not filtered_sp500.empty else 1000


max_total_count_twitter = agg_counts_bubbles_twitter['total_count'].max() if not agg_counts_bubbles_twitter.empty else 1
sizeref_calc_twitter = 2. * max_total_count_twitter / (40. ** 2)
sizeref_twitter = sizeref_calc_twitter if sizeref_calc_twitter > 0 else 1 # Sicherstellen, dass sizeref nicht 0 ist


max_total_count_news = agg_counts_bubbles_news['total_count'].max() if not agg_counts_bubbles_news.empty else 1
sizeref_calc_news = 2. * max_total_count_news / (40. ** 2)
sizeref_news = sizeref_calc_news if sizeref_calc_news > 0 else 1 # Sicherstellen, dass sizeref nicht 0 ist



if not agg_counts_bubbles_twitter.empty: # Nur hinzufügen, wenn Daten vorhanden sind
    fig.add_trace(go.Scatter(
        x=agg_counts_bubbles_twitter['quarter_start'],
        y=[bubble_y_position_twitter] * len(agg_counts_bubbles_twitter), # Konstante Y-Position
        mode='markers',
        marker=dict(
            size=agg_counts_bubbles_twitter['total_count'],
            color='green',
            sizemode='area',
            sizeref=sizeref_twitter,
            sizemin=4,
        ),
        name='Total Tweets',
        text=agg_counts_bubbles_twitter['total_count'],
        hovertemplate="Total Tweets: %{text}<extra></extra>",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=agg_counts_bubbles_twitter['quarter_start'],
        y=[bubble_y_position_twitter] * len(agg_counts_bubbles_twitter), # Konstante Y-Position
        mode='markers',
        marker=dict(
            size=agg_counts_bubbles_twitter['negative_neutral_count'],
            color='yellow',
            sizemode='area',
            sizeref=sizeref_twitter,
            sizemin=4,
        ),
        name='Neutral Tweets',
        text=agg_counts_bubbles_twitter['negative_neutral_count'],
        hovertemplate="Neutral Tweets: %{text}<extra></extra>",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=agg_counts_bubbles_twitter['quarter_start'],
        y=[bubble_y_position_twitter] * len(agg_counts_bubbles_twitter), # Konstante Y-Position
        mode='markers',
        marker=dict(
            size=agg_counts_bubbles_twitter['negative'],
            color='red',
            sizemode='area',
            sizeref=sizeref_twitter,
            sizemin=4,
        ),
        name='Negative Tweets',
        text=agg_counts_bubbles_twitter['negative'],
        hovertemplate="Negative Tweets: %{text}<extra></extra>",
        showlegend=True
    ))

# outer circle news
if not agg_counts_bubbles_news.empty: # Nur hinzufügen, wenn Daten vorhanden sind
    fig.add_trace(go.Scatter(
        x=agg_counts_bubbles_news['quarter_start'],
        y=[bubble_y_position_news] * len(agg_counts_bubbles_news), # Konstante Y-Position
        mode='markers',
        marker=dict(
            size=agg_counts_bubbles_news['total_count'],
            color='green',
            sizemode='area',
            sizeref=sizeref_news,
            sizemin=4,
        ),
        name='Total News',
        text=agg_counts_bubbles_news['total_count'],
        hovertemplate="Total News: %{text}<extra></extra>",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=agg_counts_bubbles_news['quarter_start'],
        y=[bubble_y_position_news] * len(agg_counts_bubbles_news), # Konstante Y-Position
        mode='markers',
        marker=dict(
            size=agg_counts_bubbles_news['negative_neutral_count'],
            color='yellow',
            sizemode='area',
            sizeref=sizeref_news,
            sizemin=4,
        ),
        name='Neutral News',
        text=agg_counts_bubbles_news['negative_neutral_count'],
        hovertemplate="Neutral News: %{text}<extra></extra>",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=agg_counts_bubbles_news['quarter_start'],
        y=[bubble_y_position_news] * len(agg_counts_bubbles_news), # Konstante Y-Position
        mode='markers',
        marker=dict(
            size=agg_counts_bubbles_news['negative'],
            color='red',
            sizemode='area',
            sizeref=sizeref_news,
            sizemin=4,
        ),
        name='Negative News',
        text=agg_counts_bubbles_news['negative'],
        hovertemplate="Negative News: %{text}<extra></extra>",
        showlegend=True
    ))

# --- Layout ---
fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='S&P 500', side='left'),
    yaxis2=dict(title='GDP', overlaying='y', side='right', showgrid=False),
    legend=dict(x=0.01, y=0.99), # Legendenposition
    height=600
)

# Show in Streamlit
st.plotly_chart(fig, use_container_width=True)



################################################################################
################# second plot #######################################################
################################################################################





# monthly grouped df
agg_df_twitter = agg_monthly_sent_analysis(df_twitter)
agg_df_news = agg_monthly_sent_analysis(df_news)

unique_dates_twitter = agg_df_twitter[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
unique_dates_twitter['date_order'] = unique_dates_twitter.index + 1  # 1-basiert

agg_df_twitter = agg_df_twitter.merge(unique_dates_twitter, on='date', how='left')

# show on x axis only every 6 month
tick_df_twitter = unique_dates_twitter[unique_dates_twitter['date'].dt.month.isin([1, 7])]
tickvals_twitter = tick_df_twitter['date_order'].tolist()
ticktext_twitter = tick_df_twitter['date'].dt.strftime('%b %Y').tolist()

agg_df_news = agg_df_news.merge(unique_dates_twitter, on='date', how='left')

# add source
agg_df_news['source'] = 'news'
agg_df_twitter['source'] = 'twitter'

# concat data
agg_df_comb = pd.concat([agg_df_twitter, agg_df_news], ignore_index=True)

agg_df_comb['source_sentiment'] = agg_df_comb['sentiment'] + '_' + agg_df_comb['source']

# Plot

color_map_bubbles = {
    'positive_twitter': 'green',
    'neutral_twitter': 'gray',
    'negative_twitter': 'red',
    'positive_news': 'blue',
    'neutral_news': 'orange',
    'negative_news': 'purple'
}


fig = px.scatter(
    agg_df_comb,
    x='date_order',
    y='confidence_score_mean',
    color='source_sentiment',
    size='count',
    hover_data=['count', 'date', 'confidence_score_mean', 'source'],
    animation_frame="date_order",
    color_discrete_map=color_map_bubbles,
    size_max=60,
    opacity=0.6,
    range_x=[0.5, agg_df_comb['date_order'].max() + 0.5],
)

# axis labeling and styling
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=tickvals_twitter,
        ticktext=ticktext_twitter,
        title='Date'
    ),
    yaxis=dict(
        title='Confidence Score',
        range=[0, 1],
    )
)

st.plotly_chart(fig, use_container_width=True)


############################################################################
#############################################################################
####################### TEXT and AUDIO ###################################
############################################################################
#############################################################################

# result_text = create_text_from_sent_analy_df(filtered_counts_twitter, filtered_counts_news,filtered_sp500)

# st.write(result_text)

# text = st.text_input(label='1', value=result_text)


# if st.button("Play"):
#     if isinstance(text, str) and text.strip():
#         tts = gTTS(text.strip(), lang="en")
#         tts.save("output.mp3")
#         st.audio("output.mp3", format="audio/mp3")
#     else:
#         st.warning("Textfeld ist leer oder ungültig.")

#############################################################################
