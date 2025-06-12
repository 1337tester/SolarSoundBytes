import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from streamlit_plotly_events import plotly_events


from gtts import gTTS

from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result
from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result

from solarsoundbytes.text_creation.create_text import create_text_from_sent_analy_df
from solarsoundbytes.process_sp500_df import preprocess_sp500_df
from solarsoundbytes.import_energy_data import get_energy_df
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
# counts_news = count_sent_per_quarter(df_news)
# counts_twitter = count_sent_per_quarter(df_twitter)
monthly_sp500 = preprocess_sp500_df()
df_energy = get_energy_df()

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
# filtered_sp500 = monthly_sp500[(monthly_sp500['month'] >= start_date) & (monthly_sp500['month'] <= end_date)]
# filtered_counts_twitter = counts_twitter[(counts_twitter['quarter_start'] >= start_date) & (counts_twitter['quarter_start'] <= end_date)]
# filtered_counts_news = counts_news[(counts_news['quarter_start'] >= start_date) & (counts_news['quarter_start'] <= end_date)]

# # Streamlit Selectboxes for quarter selection
# st.sidebar.header("Select Date Range")
# years = range(2022, 2024)
# quarters = ["Q1", "Q2", "Q3", "Q4"]

# start_quarter_options = [f"{year} {q}" for year in years for q in quarters]
# end_quarter_options = [f"{year} {q}" for year in years for q in quarters]

# selected_start = st.sidebar.selectbox("Start Quarter", start_quarter_options, index=start_quarter_options.index("2015 Q1"))
# selected_end = st.sidebar.selectbox("End Quarter", end_quarter_options, index=end_quarter_options.index("2024 Q4"))

# def quarter_to_dates(q_str):
#     year, q = map(int, q_str.split(" Q"))
#     start_month = (q - 1) * 3 + 1
#     end_month = start_month + 2
#     start_date = pd.to_datetime(f"{year}-{start_month:02d}-01")
#     end_date = pd.to_datetime(f"{year}-{end_month:02d}-01") + pd.offsets.MonthEnd(1)
#     return start_date, end_date

# start_date, _ = quarter_to_dates(selected_start)
# _, end_date = quarter_to_dates(selected_end)

# # Ensure start < end
# if start_date > end_date:
#     st.error("Start quarter must be before end quarter.")
#     st.stop()

# Filter datasets
filtered_sp500 = monthly_sp500[(monthly_sp500['month'] >= start_date) & (monthly_sp500['month'] <= end_date)]

# Initialize the figure
fig = go.Figure()

# --- S&P 500 Trace ---
fig.add_trace(go.Scatter(
    x=filtered_sp500['month'], y=filtered_sp500['Price'],
    name='S&P 500',
    yaxis='y1', # This will be the right y-axis (y1 is default, we'll configure it to be on the right later)
    mode='lines',
    line=dict(color='blue')
))

# --- Renewable Energy Trace ---


# Filter df_energy based on selected quarter range
df_energy['Date'] = pd.to_datetime(df_energy['Date'])
filtered_df_energy = df_energy[(df_energy['Date'] >= start_date) & (df_energy['Date'] <= end_date)]

fig.add_trace(go.Scatter(
    x=filtered_df_energy['Date'], y=filtered_df_energy['Installed Capacity'],
    name='Installed Capacity Solar + Wind (MW)',
    yaxis='y2', # This will be the right y-axis (y1 is default, we'll configure it to be on the right later)
    mode='lines',
    line=dict(color='green'),
    # visible='legendonly'
))
# --- News Sentiment Bubble Chart Trace ---
df_news['date'] = pd.to_datetime(df_news['date'])
df_news_filtered = df_news[(df_news['date'] >= start_date) & (df_news['date'] <= end_date)].copy()

# Konvertiere Datum und extrahiere Monat
df_news_filtered['date'] = pd.to_datetime(df_news_filtered['date'])
df_news_filtered['month'] = df_news_filtered['date'].dt.to_period('M').dt.to_timestamp()

# Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
df_news_filtered['correct_prob'] = df_news_filtered[['pos_score', 'neg_score']].max(axis=1)

# a
monthly_stats_news = df_news_filtered.groupby('month').agg(
    mean_correct_prob=('correct_prob', 'mean'),
    mean_pos_score=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_correct_prob=('correct_prob', 'std'),
).reset_index()

# Create scatter trace for news sentiment
# Handle potential NaN in std_correct_prob for months with only one data point
# If a month has only one news item, std dev is NaN. Set to 0 for error bar.
monthly_stats_news['std_correct_prob'] = monthly_stats_news['std_correct_prob'].fillna(0)


# --- News Sentiment Bubble Chart Trace ---
fig.add_trace(go.Scatter(
    x=monthly_stats_news['month'],
    y=monthly_stats_news['std_correct_prob'],
    mode='markers',
    marker=dict(
        size=monthly_stats_news['count'] / 3, # Using 'count' for size
        sizemode='area',
        sizeref=2. * monthly_stats_news['count'].max() / (40. ** 2),
        sizemin=4,
        color=monthly_stats_news['mean_pos_score'], # Color by mean_pos_score
        colorscale='RdYlGn',
        cmin=0.4,
        cmax=0.8,
        showscale=True,
        colorbar=dict(
            title='Mean Pos-score',
            x=0.5,
            y=1.15,
            xanchor='center',
            yanchor='top',
            orientation='h',
            len=0.5
        ),

    ),
    name='News Sentiment',
    # visible='legendonly',
    yaxis='y3', # Left y-axis
    # Customdata for hover information (excluding std_correct_prob since it's now an error bar)
    customdata=monthly_stats_news[['std_correct_prob', 'mean_pos_score']].values,
    hovertemplate=(
        "<b>Month:</b> %{x|%Y-%m}<br>" +
        "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
        "<b>Std Dev (Correct Prob):</b> %{customdata[0]:.2f}<br>" +
        "<b>Mean Pos-score:</b> %{customdata[1]:.2f}<br>" +
        "<b>News Count:</b> %{marker.size}<extra></extra>"
    )
))

# --- NEW: Separate Trace for Standard Deviation Error Bars ---
fig.add_trace(go.Scatter(
    x=monthly_stats_news['month'],
    y=monthly_stats_news['std_correct_prob'], # Y-values are the same as the bubbles
    mode='lines', # No markers or lines for this trace, only error bars
    line=dict(
        color='grey',
        width=0,
    ),
    error_y=dict(
        type='data',
        array=monthly_stats_news['std_correct_prob'],
        symmetric=True,
        visible=True,
        color='grey', # You can choose a color for your error bars
        width=1 # Thickness of the error bar line
    ),
    name='Std (News Sentiment)', # Name for the legend
    yaxis='y3', # Use the same y-axis as the bubbles
    showlegend=True, # Ensure it shows in the legend
    hoverinfo='skip', # Skip hover info for this trace to avoid clutter
    visible='legendonly'
))

####################
########Twitter
################

df_twitter['date'] = pd.to_datetime(df_twitter['date'])
df_twitter_filtered = df_twitter[(df_twitter['date'] >= start_date) & (df_twitter['date'] <= end_date)].copy()

# Konvertiere Datum und extrahiere Monat
df_twitter_filtered['date'] = pd.to_datetime(df_twitter_filtered['date'])



df_twitter_filtered['month'] = df_twitter_filtered['date'].dt.to_period('M').dt.to_timestamp()

# Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
df_twitter_filtered['correct_prob'] = df_twitter_filtered[['pos_score', 'neg_score']].max(axis=1)

# a
monthly_stats_twitter = df_twitter_filtered.groupby('month').agg(
    mean_correct_prob=('correct_prob', 'mean'),
    mean_pos_score=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_correct_prob=('correct_prob', 'std'),
).reset_index()

# Create scatter trace for news sentiment
# Handle potential NaN in std_correct_prob for months with only one data point
# If a month has only one news item, std dev is NaN. Set to 0 for error bar.colorscale='RdYlGn',
monthly_stats_twitter['std_correct_prob'] = monthly_stats_twitter['std_correct_prob'].fillna(0)

# Generate random offsets for y-values

# --- News Sentiment Bubble Chart Trace (main trace) ---
fig.add_trace(go.Scatter(
    x=monthly_stats_twitter['month'], # X-Achse
    y=monthly_stats_twitter['std_correct_prob'], # **NEU: Y-Achse ist jetzt die Standardabweichung**
    mode='markers',
    marker=dict(
        symbol='diamond',  # Markerform
        size=monthly_stats_twitter['count']/3, # Größe anpassen
        sizemode='area',
        sizeref=2. * monthly_stats_twitter['count'].max() / (40. ** 2), # sizeref anpassen
        sizemin=4, # sizemin anpassen
        color=monthly_stats_twitter['mean_pos_score'], # Farbe weiterhin nach Pos-Score
        colorscale='RdYlGn',        cmin=0,
        cmax=1,
        showscale=False, # Colorbar für Twitter Sentiment könnte dupliziert sein, wenn News sie schon hat
    ),
    name='Twitter Sentiment',
    yaxis='y3', # Nutzt die y3-Achse
    customdata=monthly_stats_twitter[['mean_pos_score', 'std_correct_prob']].values,
    hovertemplate=(
        "<b>Month:</b> %{x|%Y-%m}<br>" +
        "<b>Std Dev (Correct Prob):</b> %{y:.2f}<br>" + # Y ist jetzt std dev
        "<b>Mean Pos-score:</b> %{customdata[0]:.2f}<br>" +
        "<b>Twitter Count:</b> %{marker.size}<extra></extra>"
    )
))

# --- Update Layout ---
fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(
        title='S&P 500 ($)',
        side='right', # S&P 500 on the right
        #position = 0.95,
        showgrid=False
    ),
    yaxis2=dict(
        title='Installed Capacity Solar + Wind (MW)',
        side='right', # Secondary right axis
        overlaying='y', # Overlays the primary y-axis
        anchor='free',  # Allows it to be positioned independently
        autoshift=True, # THIS IS THE KEY! Automatically shifts to avoid overlap
        showgrid=False,
        automargin=True # Let Plotly adjust margin for this axis if needed

    ),
    yaxis3=dict(
        title='Mean Probability of Correct Sentiment (%)',
        side='left', # News Sentiment on the left
        showgrid=True,
        anchor='free', # Allow free positioning
        overlaying='y', # Overlay on the primary y-axis
        autorange="reversed",
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

#######################Switch to Events Site ################################
############################################################################
if st.button("Russias invasion of Ukraine "):
    st.switch_page("pages/event_russia.py")

if st.button("REPowerEU"):
    st.switch_page("pages/repowereu.py")

if st.button("COP28"):
    st.switch_page("pages/cop28.py")

if st.button("Solar Invest > Oil Invest"):
    st.switch_page("pages/solar_oil.py")

if st.button("IRA USA"):
    st.switch_page("pages/eu_ira.py")

if st.button("Solar Production > 1 TW"):
    st.switch_page("pages/solar_1tw.py")


############################################################################
#############################################################################
####################### TEXT and AUDIO ###################################
############################################################################
#############################################################################

result_text = create_text_from_sent_analy_df(monthly_stats_twitter, monthly_stats_news ,filtered_sp500, filtered_df_energy)

st.write(result_text)

# text = st.text_input(label='1', value=result_text)


# if st.button("Play"):
if isinstance(result_text, str) and result_text.strip():
        tts = gTTS(result_text.strip(), lang="en")
        tts.save("output.mp3")
        st.audio("output.mp3", format="audio/mp3")
else:
        st.warning("Text field is empty or invalid.")

#############################################################################




#####################################################################
####################################################################
#####################################################################
###################################################################

# # --- Plot- ---
# fig = go.Figure()

# # S&P 500 line
# fig.add_trace(go.Scatter(
#     x=filtered_sp500['month'], y=filtered_sp500['Price'],
#     name='S&P 500',
#     yaxis='y1',
#     mode='lines+markers',
#     line=dict(color='blue')
# ))

# df_energy['bar_center'] = pd.to_datetime(df_energy['year'].astype(str)) + pd.DateOffset(months=6)

# # Renewable Energies installed Capacity in MW

# bar_width_ms = 365 * 24 * 60 * 60 * 1000

# bar_halfwidth = bar_width_ms / 2

# x_start = df_energy['bar_center'] - pd.to_timedelta(bar_halfwidth, unit='ms')
# x_end   = df_energy['bar_center'] + pd.to_timedelta(bar_halfwidth, unit='ms')
# y_top   = df_energy['renewables']
# x_lines = []
# y_lines = []

# for xs, xe, y in zip(x_start, x_end, y_top):
#     x_lines += [xs, xe, None]
#     y_lines += [y, y, None]

# fig.add_trace(go.Scatter(
#     x=x_lines,
#     y=y_lines,
#     mode='lines',
#     yaxis='y2',
#     name='renewable energy',
#     line=dict(color='orange', width=3),
#     showlegend=True,
#     hoverinfo='skip'
# ))
# GDP line
# # first_gdp = True

# x_all = []
# y_all = []

# for _, row in df_energy.iterrows():
#     year = int(row['year'])
#     value = row['renewables']

#     year_start = pd.to_datetime(f"{year}-01-01")
#     year_end = pd.to_datetime(f"{year}-12-31")

#     start_q_year, start_q = map(int, selected_start.split(" Q"))

#     if year < start_q_year:
#         continue

#     if year == start_q_year:
#         start_month = (start_q - 1) * 3 + 1
#         bar_start = pd.to_datetime(f"{year}-{start_month:02d}-01")
#     else:
#         bar_start = year_start

#     # Linie von bar_start bis year_end auf Höhe value
#     x_all.extend([bar_start, year_end, None])  # None trennt Linienstücke
#     y_all.extend([value, value, None])

# fig.add_trace(go.Scatter(
#     x=x_all,
#     y=y_all,
#     mode='lines',
#     line=dict(color='orange', width=4),
#     name='renewable energy',
#     yaxis='y2'
# ))





# agg_counts_bubbles_twitter = filtered_counts_twitter.pivot_table(
#     index='quarter_start',
#     columns='sentiment',
#     values='count',
#     fill_value=0
# ).reset_index()

# agg_counts_bubbles_news = filtered_counts_news.pivot_table(
#     index='quarter_start',
#     columns='sentiment',
#     values='count',
#     fill_value=0
# ).reset_index()


# agg_counts_bubbles_twitter['total_count'] = agg_counts_bubbles_twitter['positive'] + agg_counts_bubbles_twitter['neutral'] + agg_counts_bubbles_twitter['negative']
# agg_counts_bubbles_twitter['negative_neutral_count'] = agg_counts_bubbles_twitter['neutral'] + agg_counts_bubbles_twitter['negative']
# agg_counts_bubbles_news['total_count'] = agg_counts_bubbles_news['positive']    # + agg_counts_bubbles_news['neutral'] + agg_counts_bubbles_news['negative']
# agg_counts_bubbles_news['negative_neutral_count'] = 0   #agg_counts_bubbles_news['neutral'] + agg_counts_bubbles_news['negative']


# bubble_y_position_twitter = filtered_sp500['Price'].min() - 2000 if not filtered_sp500.empty else 1000
# bubble_y_position_news = filtered_sp500['Price'].min() - 1500 if not filtered_sp500.empty else 1000


# max_total_count_twitter = agg_counts_bubbles_twitter['total_count'].max() if not agg_counts_bubbles_twitter.empty else 1
# sizeref_calc_twitter = 2. * max_total_count_twitter / (40. ** 2)
# sizeref_twitter = sizeref_calc_twitter if sizeref_calc_twitter > 0 else 1 # Sicherstellen, dass sizeref nicht 0 ist


# max_total_count_news = agg_counts_bubbles_news['total_count'].max() if not agg_counts_bubbles_news.empty else 1
# sizeref_calc_news = 2. * max_total_count_news / (40. ** 2)
# sizeref_news = sizeref_calc_news if sizeref_calc_news > 0 else 1 # Sicherstellen, dass sizeref nicht 0 ist



# if not agg_counts_bubbles_twitter.empty: # Nur hinzufügen, wenn Daten vorhanden sind
#     fig.add_trace(go.Scatter(
#         x=agg_counts_bubbles_twitter['quarter_start'],
#         y=[bubble_y_position_twitter] * len(agg_counts_bubbles_twitter), # Konstante Y-Position
#         mode='markers',
#         marker=dict(
#             size=agg_counts_bubbles_twitter['total_count'],
#             color='green',
#             sizemode='area',
#             sizeref=sizeref_twitter,
#             sizemin=4,
#         ),
#         name='Total Tweets',
#         text=agg_counts_bubbles_twitter['total_count'],
#         hovertemplate="Total Tweets: %{text}<extra></extra>",
#         showlegend=True
#     ))

#     fig.add_trace(go.Scatter(
#         x=agg_counts_bubbles_twitter['quarter_start'],
#         y=[bubble_y_position_twitter] * len(agg_counts_bubbles_twitter), # Konstante Y-Position
#         mode='markers',
#         marker=dict(
#             size=agg_counts_bubbles_twitter['negative_neutral_count'],
#             color='yellow',
#             sizemode='area',
#             sizeref=sizeref_twitter,
#             sizemin=4,
#         ),
#         name='Neutral Tweets',
#         text=agg_counts_bubbles_twitter['negative_neutral_count'],
#         hovertemplate="Neutral Tweets: %{text}<extra></extra>",
#         showlegend=True
#     ))

#     fig.add_trace(go.Scatter(
#         x=agg_counts_bubbles_twitter['quarter_start'],
#         y=[bubble_y_position_twitter] * len(agg_counts_bubbles_twitter), # Konstante Y-Position
#         mode='markers',
#         marker=dict(
#             size=agg_counts_bubbles_twitter['negative'],
#             color='red',
#             sizemode='area',
#             sizeref=sizeref_twitter,
#             sizemin=4,
#         ),
#         name='Negative Tweets',
#         text=agg_counts_bubbles_twitter['negative'],
#         hovertemplate="Negative Tweets: %{text}<extra></extra>",
#         showlegend=True
#     ))

# # outer circle news
# if not agg_counts_bubbles_news.empty: # Nur hinzufügen, wenn Daten vorhanden sind
#     fig.add_trace(go.Scatter(
#         x=agg_counts_bubbles_news['quarter_start'],
#         y=[bubble_y_position_news] * len(agg_counts_bubbles_news), # Konstante Y-Position
#         mode='markers',
#         marker=dict(
#             size=agg_counts_bubbles_news['total_count'],
#             color='green',
#             sizemode='area',
#             sizeref=sizeref_news,
#             sizemin=4,
#         ),
#         name='Total News',
#         text=agg_counts_bubbles_news['total_count'],
#         hovertemplate="Total News: %{text}<extra></extra>",
#         showlegend=True
#     ))

#     fig.add_trace(go.Scatter(
#         x=agg_counts_bubbles_news['quarter_start'],
#         y=[bubble_y_position_news] * len(agg_counts_bubbles_news), # Konstante Y-Position
#         mode='markers',
#         marker=dict(
#             size=agg_counts_bubbles_news['negative_neutral_count'],
#             color='yellow',
#             sizemode='area',
#             sizeref=sizeref_news,
#             sizemin=4,
#         ),
#         name='Neutral News',
#         text=agg_counts_bubbles_news['negative_neutral_count'],
#         hovertemplate="Neutral News: %{text}<extra></extra>",
#         showlegend=True
#     ))

#     fig.add_trace(go.Scatter(
#         x=agg_counts_bubbles_news['quarter_start'],
#         y=[bubble_y_position_news] * len(agg_counts_bubbles_news), # Konstante Y-Position
#         mode='markers',
#         marker=dict(
#             size=agg_counts_bubbles_news['negative'],
#             color='red',
#             sizemode='area',
#             sizeref=sizeref_news,
#             sizemin=4,
#         ),
#         name='Negative News',
#         text=agg_counts_bubbles_news['negative'],
#         hovertemplate="Negative News: %{text}<extra></extra>",
#         showlegend=True
#     ))

# --- Layout ---
# fig.update_layout(
#     xaxis=dict(title='Date'),
#     yaxis=dict(title='S&P 500', side='left'),
#     yaxis2=dict(title='Installed electrical capacity renewable energy (MW)', overlaying='y', side='right', showgrid=False),
#     legend=dict(x=0.01, y=0.99), # Legendenposition
#     height=600
# )

# # Show in Streamlit
# st.plotly_chart(fig, use_container_width=True)



################################################################################
################# second plot #######################################################
################################################################################

# df = df_news


# # Konvertiere Datum und extrahiere Monat
# df['date'] = pd.to_datetime(df['date'])
# df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

# # Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
# df['correct_prob'] = df[['pos_score', 'neg_score']].max(axis=1)

# # Aggregiere nach Monat
# monthly_stats = df.groupby('month').agg(
#     mean_correct_prob=('correct_prob', 'mean'),
#     mean_pos_score=('pos_score', 'mean'),
#     count=('correct_prob', 'count')
# ).reset_index()

# # Bubble Chart
# fig = px.scatter(
#     monthly_stats,
#     x='month',
#     y='mean_correct_prob',  # y-Achse = mean probability of correct sentiment
#     size='count',            # Größe = Anzahl der News
#     color='mean_pos_score',  # Farbe = mittlerer pos_score (Stimmung)
#     color_continuous_scale='RdYlGn',  # Rot (negativ) bis Grün (positiv)
#     #range_color=[0, 1],
#     labels={
#         'month': 'Month',
#         'mean_correct_prob': 'Mean Probability of Correct Sentiment',
#         'count': 'Number News',
#         'mean_pos_score': 'Mean Pos-score'
#     },

# )

# # Größe feintunen
# fig.update_traces(marker=dict(
#     sizemode='area',
#     sizeref=2. * monthly_stats['count'].max() / (40. ** 2),
#     sizemin=4,

#     ),
# )

# x_all = []
# y_all = []

# for _, row in df_energy.iterrows():
#     year = int(row['year'])
#     value = row['renewables']

#     year_start = pd.to_datetime(f"{year}-01-01")
#     year_end = pd.to_datetime(f"{year}-12-31")

#     start_q_year, start_q = map(int, selected_start.split(" Q"))

#     if year < start_q_year:
#         continue

#     if year == start_q_year:
#         start_month = (start_q - 1) * 3 + 1
#         bar_start = pd.to_datetime(f"{year}-{start_month:02d}-01")
#     else:
#         bar_start = year_start

#     # Linie von bar_start bis year_end auf Höhe value
#     x_all.extend([bar_start, year_end, None])  # None trennt Linienstücke
#     y_all.extend([value, value, None])

# fig.add_trace(go.Scatter(
#     x=x_all,
#     y=y_all,
#     mode='lines',
#     line=dict(color='orange', width=4),
#     name='renewable energy',
#     yaxis='y2'
# ))
# fig.add_trace(go.Scatter(
#     x=filtered_sp500['month'], y=filtered_sp500['Price'],
#     name='S&P 500',
#     yaxis='y3',  # dritte y-Achse
#     mode='lines+markers',
#     line=dict(color='blue')
# ))

# # Layout: Zwei Y-Achsen (links und rechts)
# fig.update_layout(
#     #xaxis=dict(title='Zeit'),
#     yaxis=dict(
#         title='Correct Sentiment Probability',
#         side='left'
#     ),
#     yaxis2=dict(
#         title='Zweite rechte Y-Achse',
#         overlaying='y',
#         side='right',
#         position=0.95
#     ),
#     yaxis3=dict(
#         title='S&P 500',
#         overlaying='y',
#         side='right',
#         position=0.85
#     ),
#     xaxis=dict(
#         domain=[0, 0.8]  # Bereich für x-Achse etwas kleiner
#     )
# )

# # In Streamlit anzeigen
# st.plotly_chart(fig, use_container_width=True)







# # monthly grouped df
# agg_df_twitter = agg_monthly_sent_analysis(df_twitter)
# agg_df_news = agg_monthly_sent_analysis(df_news)
# st.write(agg_df_news.head())

# unique_dates_twitter = agg_df_twitter[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
# unique_dates_twitter['date_order'] = unique_dates_twitter.index + 1  # 1-basiert

# agg_df_twitter = agg_df_twitter.merge(unique_dates_twitter, on='date', how='left')

# # show on x axis only every 6 month
# tick_df_twitter = unique_dates_twitter[unique_dates_twitter['date'].dt.month.isin([1, 7])]
# tickvals_twitter = tick_df_twitter['date_order'].tolist()
# ticktext_twitter = tick_df_twitter['date'].dt.strftime('%b %Y').tolist()

# agg_df_news = agg_df_news.merge(unique_dates_twitter, on='date', how='left')

# # add source
# agg_df_news['source'] = 'news'
# agg_df_twitter['source'] = 'twitter'

# # concat data
# agg_df_comb = pd.concat([agg_df_twitter, agg_df_news], ignore_index=True)

# agg_df_comb['source_sentiment'] = agg_df_comb['sentiment'] + '_' + agg_df_comb['source']

# # Plot

# color_map_bubbles = {
#     'positive_twitter': 'green',
#     'neutral_twitter': 'gray',
#     'negative_twitter': 'red',
#     'positive_news': 'blue',
#     'neutral_news': 'orange',
#     'negative_news': 'purple'
# }


# fig = px.scatter(
#     agg_df_comb,
#     x='date_order',
#     y='confidence_score_mean',
#     color='source_sentiment',
#     size='count',
#     hover_data=['count', 'date', 'confidence_score_mean', 'source'],
#     animation_frame="date_order",
#     color_discrete_map=color_map_bubbles,
#     size_max=60,
#     opacity=0.6,
#     range_x=[0.5, agg_df_comb['date_order'].max() + 0.5],
# )

# # axis labeling and styling
# fig.update_layout(
#     xaxis=dict(
#         tickmode='array',
#         tickvals=tickvals_twitter,
#         ticktext=ticktext_twitter,
#         title='Date'
#     ),
#     yaxis=dict(
#         title='Confidence Score',
#         range=[0, 1],
#     )
# )

# st.plotly_chart(fig, use_container_width=True)
