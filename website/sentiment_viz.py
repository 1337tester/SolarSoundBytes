import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result, create_df_of_twitter_result_events
from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result
from solarsoundbytes.process_sp500_df import preprocess_sp500_df
from solarsoundbytes.import_energy_data import get_energy_df


# --- Streamlit UI ---
st.set_page_config(page_title="Monthly Sentiment Visualization", layout="wide")
st.title("Monthly Sentiment Consensus: Articles vs Tweets")

# --- Generate mock data ---
def generate_mock_data():
    months = pd.date_range('2022-01-01', '2024-12-01', freq='MS')
    data = []
    for source in ['article', 'tweet']:
        for month in months:
            mean_sentiment = np.random.uniform(-1, 1)
            std_sentiment = np.random.uniform(0.1, 0.5)
            count = np.random.randint(10, 200)
            data.append({
                'month': month,
                'source': source,
                'mean_sentiment': mean_sentiment,
                'std_sentiment': std_sentiment,
                'count': count
            })
    return pd.DataFrame(data)

df_news = create_df_of_newsarticle_result()
df_news['date'] = pd.to_datetime(df_news['date'])

# Konvertiere Datum und extrahiere Monat
df_news['date'] = pd.to_datetime(df_news['date'])
df_news['month'] = df_news['date'].dt.to_period('M').dt.to_timestamp()
# Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
df_news['correct_prob'] = df_news[['pos_score', 'neg_score']].max(axis=1)

# a
monthly_stats_news = df_news.groupby('month').agg(
    mean_sentiment=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_sentiment=('correct_prob', 'std'),
).reset_index()
monthly_stats_news['source'] = 'article'


df_twitter = create_df_of_twitter_result()
df_twitter['date'] = pd.to_datetime(df_twitter['date'])

# Konvertiere Datum und extrahiere Monat
df_twitter['date'] = pd.to_datetime(df_twitter['date'])
df_twitter['month'] = df_twitter['date'].dt.to_period('M').dt.to_timestamp()

# Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
df_twitter['correct_prob'] = df_twitter[['pos_score', 'neg_score']].max(axis=1)

# a
monthly_stats_twitter = df_twitter.groupby('month').agg(
    mean_sentiment=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_sentiment=('correct_prob', 'std'),
).reset_index()
monthly_stats_twitter['source'] = 'tweet'


df = pd.concat([monthly_stats_twitter, monthly_stats_news])


df_twitter_events = create_df_of_twitter_result_events()
df_twitter_events['date'] = pd.to_datetime(df_twitter_events['date'])

df_twitter_events['date'] = pd.to_datetime(df_twitter_events['date'])
df_twitter_events['hour'] = df_twitter_events['date'].dt.to_period('H').dt.to_timestamp()

df_twitter_events['correct_prob'] = df_twitter_events[['pos_score', 'neg_score']].max(axis=1)

monthly_stats_twitter_events = df_twitter_events.groupby('hour').agg(
    mean_sentiment=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_sentiment=('correct_prob', 'std'),
).reset_index()
monthly_stats_twitter_events['source'] = 'tweet'

def sentiment_color(val):
    # Red (-1) to Green (+1)
    r = int(255 * (1 - (val + 1) / 2))
    g = int(255 * ((val + 1) / 2))
    return f'rgb({r},{g},100)'

# --- Global Events Data ---
GLOBAL_EVENTS = {
    "Russian invasion of Ukraine": "2022-02-24",
    "EU announces REPowerEU plan": "2022-05-18",
    "US Inflation Reduction Act signed (major climate/energy provisions)": "2022-08-16",
    "IEA: global solar power generation surpasses oil for the first time": "2023-04-20",
    "Global installed solar PV capacity surpasses 1 terawatt milestone": "2023-12-30",
    "COP28 concludes with historic agreement to transition away from fossil fuels": "2023-12-13"
}

# Convert event dates to datetime objects
EVENT_DATES = {event: pd.to_datetime(date) for event, date in GLOBAL_EVENTS.items()}

df_news['date'] = pd.to_datetime(df_news['date'])

event_dates_list = list(EVENT_DATES.values())

event_dates_only_day = [d.date() for d in event_dates_list]
extended_event_dates_only_day = set()
for event_date in event_dates_list:
    # Fügen Sie das Event-Datum selbst hinzu
    extended_event_dates_only_day.add(event_date.date())
    # Fügen Sie den Tag davor hinzu
    extended_event_dates_only_day.add((event_date - timedelta(days=1)).date())
    # Fügen Sie den Tag danach hinzu
    extended_event_dates_only_day.add((event_date + timedelta(days=1)).date())

df_news_filtered_by_event_dates_day_only = df_news[df_news['date'].dt.date.isin(extended_event_dates_only_day)]
df_news_filtered_by_event_dates_day_only['hour'] = df_news_filtered_by_event_dates_day_only['date'].dt.to_period('H').dt.to_timestamp()

monthly_stats_news_events = df_news_filtered_by_event_dates_day_only.groupby('hour').agg(
    mean_sentiment=('pos_score', 'mean'),
    count=('correct_prob', 'count'),
    std_sentiment=('correct_prob', 'std'),
).reset_index()
monthly_stats_news_events['source'] = 'article'


# Use df_events for the combined hourly event data
df_events = pd.concat([monthly_stats_twitter_events, monthly_stats_news_events])


# --- Short Event Labels for Annotations ---
SHORT_EVENT_LABELS = {
    "Russian invasion of Ukraine": "Russia Invasion",
    "EU announces REPowerEU plan": "REPowerEU Plan",
    "US Inflation Reduction Act signed (major climate/energy provisions)": "US IRA Signed",
    "IEA: global solar power generation surpasses oil for the first time": "Solar>Oil IEA",
    "Global installed solar PV capacity surpasses 1 terawatt milestone": "1TW Solar PV",
    "COP28 concludes with historic agreement to transition away from fossil fuels": "COP28 Fossil Fuels"
}

# --- Generate mock metric data ---
def generate_metric_data(months):
    metrics = {
        'Solar Investment': np.random.normal(100, 10, len(months)),
        'Oil Investment': np.random.normal(80, 15, len(months)),
        'Renewable Energy Jobs': np.random.normal(50, 5, len(months)),
        'Carbon Emissions': np.random.normal(200, 20, len(months))
    }
    return pd.DataFrame({
        'month': months,
        **metrics
    })

# --- UI Description ---
st.markdown("""
- **Circles**: Articles (Monthly)
- **Rhombi**: Tweets (Monthly)
- **Stars**: Articles/Tweets (Hourly Event data) - Only visible when an event is selected.
- **Y-axis**: Sentiment consensus (higher = more agreement, lower = more disagreement)
- **Color**: Red (negative) to Green (positive)
- **Size**: Number of texts (articles/tweets)
""")

# Controls
months = df['month'].dt.strftime('%Y-%m').unique() # Use months from combined df

# Event Selection
st.sidebar.header("Event Selection")
selected_event = st.sidebar.selectbox(
    "Select Global Event (shows hourly data for +/- 1 day):",
    options=["None"] + [f"{date.strftime('%Y-%m-%d')} {event}" for event, date in EVENT_DATES.items()]
)

# --- Time Window Selection & Data Preparation ---
df_window = pd.DataFrame(columns=df.columns) # Initialize empty for monthly data
df_window_events = pd.DataFrame(columns=df_events.columns) # Initialize empty for hourly data

if selected_event and selected_event != "None":
    event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
    event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])

    # Define the 3-day window for hourly data
    start_hourly_window = event_date - pd.Timedelta(days=1)
    end_hourly_window = event_date + pd.Timedelta(days=1)

    # Filter df_events for the selected 3-day window
    df_window_events = df_events[
        (df_events['hour'] >= start_hourly_window) &
        (df_events['hour'] <= end_hourly_window)
    ].copy()


    # df_window remains empty, so no monthly data points are added for sentiment

else:
    # If no event is selected, use the slider for monthly data
    months_for_slider = df['month'].unique() # All unique months from the combined df
    start_idx, end_idx = st.select_slider(
        "Select time window:",
        options=list(range(len(months_for_slider))),
        value=(0, len(months_for_slider)-1),
        format_func=lambda x: months_for_slider[x].strftime('%Y-%m')
    )
    df_window = df[(df['month'] >= months_for_slider[start_idx]) & (df['month'] <= months_for_slider[end_idx])].copy()

# Animation Controls - Animation is currently only for monthly data
st.sidebar.header("Animation Controls")
# Animation is not compatible with the hourly event view in its current form, disable if event selected
show_animation = st.sidebar.checkbox("Show animation (month by month)", value=False,
                                    disabled=(selected_event and selected_event != "None"))
if selected_event and selected_event != "None":
    st.sidebar.info("Animation is disabled when an event is selected, as it focuses on hourly data.")


# Metric Selection
st.sidebar.header("Metric Overlays")
selected_metrics = st.sidebar.multiselect(
    "Select metrics to overlay:",
    options=['S&P 500', 'Installed Capacity Renewables'],
    default=[]
)

# Generate metric data (always for the full monthly range initially)
monthly_sp500 = preprocess_sp500_df()
df_energy = get_energy_df()
df_energy = df_energy.rename(columns={'Date': 'month'})
metric_df = pd.merge(monthly_sp500, df_energy, on='month', how='outer')
metric_df = metric_df.rename(columns={'Price': 'S&P 500', 'Installed Capacity': 'Installed Capacity Renewables'})

# --- Plotly Figure ---
fig = go.Figure()

# Define sentiment traces to plot
sentiment_traces_config = []

if selected_event and selected_event != "None":
    # If an event is selected, add hourly event traces for both article and tweet
    if not df_window_events.empty:
        # Note: 'article' and 'tweet' sources are now filtered from df_events
        sentiment_traces_config.append(('article_hourly_event', 'circle', 'Article (Hourly Event)', 'article_hourly_group'))
        sentiment_traces_config.append(('tweet_hourly_event', 'diamond', 'Tweet (Hourly Event)', 'tweet_hourly_group'))
else:
    # If no event is selected, add monthly article and tweet traces
    sentiment_traces_config.append(('article', 'circle', 'Article (Monthly)', 'article_group'))
    sentiment_traces_config.append(('tweet', 'diamond', 'Tweet (Monthly)', 'tweet_monthly_group'))

# 1. Initialize sentiment traces as placeholders
for source_name_for_trace, shape, display_name, legend_group in sentiment_traces_config:
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers',
        marker=dict(
            size=[],
            color=[],
            symbol=shape,
            line=dict(width=1, color='black'),
            opacity=1.0
        ),
        name=display_name,
        legendgroup=legend_group,
        showlegend=True,
        text=[],
        hoverinfo='text',
    ))

# Define colors for metric overlays early
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

# Add metric traces
# Metrics should always show the full range unless explicitly filtered (which isn't implemented for metrics here)
# This loop for metrics assumes they are always added and not affected by animation state for now.
for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
    trace_yaxis_value = f'y{metric_idx + 2}'
    fig.add_trace(go.Scatter(
        x=metric_df['month'], # Metrics always use monthly data (full range)
        y=metric_df[metric],
        mode='lines',
        name=metric,
        line=dict(color=color),
        yaxis=trace_yaxis_value,
        showlegend=True,
        legendgroup=metric
    ))

# 2. Populate the initial traces (static view)
if not show_animation:
    # Removed trace_idx_counter here, as we directly address fig.data[0] and fig.data[1] in the event case

    if selected_event and selected_event != "None":
        # Populate hourly event data for both article and tweet (using df_window_events)
        # Trace indices are now 0 for Article and 1 for Tweet based on initialization above

        # Article Hourly Event - Corresponds to fig.data[0]
        hourly_data_article = df_window_events[df_window_events['source'] == 'article']
        if not hourly_data_article.empty:
            fig.data[0].x = hourly_data_article['hour'].tolist()
            fig.data[0].y = hourly_data_article['std_sentiment'].tolist()
            # Temporär die Größe stark erhöhen, um Sichtbarkeit zu testen
            fig.data[0].marker.size = np.sqrt(hourly_data_article['count']) * 5 + 10 # Beispiel: *5 + 10px Minimum
            fig.data[0].marker.color = [sentiment_color(v) for v in hourly_data_article['mean_sentiment']]
            fig.data[0].text = [
                f"Article (Hourly Event)<br>Hour: {h.strftime('%Y-%m-%d %H:%M')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
                for h, s, c, cnt in zip(hourly_data_article['hour'], hourly_data_article['mean_sentiment'], hourly_data_article['std_sentiment'], hourly_data_article['count'])
            ]

        # Tweet Hourly Event - Corresponds to fig.data[1]
        hourly_data_tweet = df_window_events[df_window_events['source'] == 'tweet']
        if not hourly_data_tweet.empty:
            fig.data[1].x = hourly_data_tweet['hour'].tolist()
            fig.data[1].y = hourly_data_tweet['std_sentiment'].tolist()
            # Temporär die Größe stark erhöhen, um Sichtbarkeit zu testen
            fig.data[1].marker.size = np.sqrt(hourly_data_tweet['count']) + 20 # Beispiel: *5 + 10px Minimum
            fig.data[1].marker.color = [sentiment_color(v) for v in hourly_data_tweet['mean_sentiment']]
            fig.data[1].text = [
                f"Tweet (Hourly Event)<br>Hour: {h.strftime('%Y-%m-%d %H:%M')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
                for h, s, c, cnt in zip(hourly_data_tweet['hour'], hourly_data_tweet['mean_sentiment'], hourly_data_tweet['std_sentiment'], hourly_data_tweet['count'])
            ]

    elif not df_window.empty: # Populate monthly sentiment data if no event is selected
        # Trace indices are 0 for Article and 1 for Tweet based on initialization above

        # Monthly Article
        d_all_data_article = df_window[df_window['source'] == 'article']
        if not d_all_data_article.empty:
            fig.data[0].x = d_all_data_article['month'].tolist()
            fig.data[0].y = d_all_data_article['std_sentiment'].tolist()
            fig.data[0].marker.size = np.sqrt(d_all_data_article['count']) * 3
            fig.data[0].marker.color = [sentiment_color(v) for v in d_all_data_article['mean_sentiment']]
            fig.data[0].text = [
                f"Article (Monthly)<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
                for m, s, c, cnt in zip(d_all_data_article['month'], d_all_data_article['mean_sentiment'], d_all_data_article['std_sentiment'], d_all_data_article['count'])
            ]

        # Monthly Tweet
        d_all_data_tweet = df_window[df_window['source'] == 'tweet']
        if not d_all_data_tweet.empty:
            fig.data[1].x = d_all_data_tweet['month'].tolist()
            fig.data[1].y = d_all_data_tweet['std_sentiment'].tolist()
            fig.data[1].marker.size = np.sqrt(d_all_data_tweet['count']) * 3 * 0.1 # Diamond size adjusted
            fig.data[1].marker.color = [sentiment_color(v) for v in d_all_data_tweet['mean_sentiment']]
            fig.data[1].text = [
                f"Tweet (Monthly)<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
                for m, s, c, cnt in zip(d_all_data_tweet['month'], d_all_data_tweet['mean_sentiment'], d_all_data_tweet['std_sentiment'], d_all_data_tweet['count'])
            ]

    # elif not df_window.empty: # Populate monthly sentiment data if no event is selected
    #     for source_name_in_df, shape, display_name, legend_group in [('article', 'circle', 'Article (Monthly)', 'article_group'), ('tweet', 'diamond', 'Tweet (Monthly)', 'tweet_monthly_group')]:
    #         d_all_data = df_window[df_window['source'] == source_name_in_df]
    #         if not d_all_data.empty:
    #             fig.data[trace_idx_counter].x = d_all_data['month'].tolist()
    #             fig.data[trace_idx_counter].y = d_all_data['std_sentiment'].tolist()
    #             base_size = np.sqrt(d_all_data['count']) * 3
    #             if shape == 'diamond':
    #                 fig.data[trace_idx_counter].marker.size = base_size * 0.1
    #             else:
    #                 fig.data[trace_idx_counter].marker.size = base_size
    #             fig.data[trace_idx_counter].marker.color = [sentiment_color(v) for v in d_all_data['mean_sentiment']]
    #             fig.data[trace_idx_counter].text = [
    #                 f"{display_name}<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
    #                 for m, s, c, cnt in zip(d_all_data['month'], d_all_data['mean_sentiment'], d_all_data['std_sentiment'], d_all_data['count'])
    #             ]
    #         trace_idx_counter += 1
    # else: if both empty, no traces will be populated, resulting in an empty plot.

# Prepare dynamic y-axis definitions and trace assignments for metrics
metric_yaxis_layout_config = {}
current_plot_yaxis_id = 2 # Start from y2 for metrics

# This loop ensures that the Y-axis config for metrics is always set up,
# regardless of whether metrics are selected, as it's needed for the layout.
for metric_idx, (metric, color) in enumerate(zip(['S&P 500', 'Installed Capacity Renewables'], colors)): # Use all possible metrics for config
    layout_axis_key = f'yaxis{current_plot_yaxis_id}'
    metric_yaxis_layout_config[layout_axis_key] = dict(
        title=metric,
        overlaying='y',
        side='right',
        showgrid=False,
        automargin=True,
        anchor='free',
        position=1 - (0.07 * (current_plot_yaxis_id - 1))
    )
    current_plot_yaxis_id += 1


# --- Update Layout and X-Axis based on selected_event ---
x_axis_tickformat = '%Y-%m' # Default for monthly view
x_min_display = None
x_max_display = None

if selected_event and selected_event != "None":
    x_axis_tickformat = '%Y-%m-%d %H:%M' # Detailed format for hourly view

    if not df_window_events.empty:
        x_min_display = df_window_events['hour'].min() - pd.Timedelta(hours=2) # Add some padding
        x_max_display = df_window_events['hour'].max() + pd.Timedelta(hours=2) # Add some padding
    else:
        # Fallback if no hourly data for the event (shouldn't happen if data exists for selected event)
        event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
        event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])
        x_min_display = event_date - pd.Timedelta(days=1, hours=2)
        x_max_display = event_date + pd.Timedelta(days=1, hours=2)

else:
    # Use the selected slider range for monthly view
    months_for_slider = df['month'].unique() # Re-get months for full range
    if not df_window.empty:
        x_min_display = df_window['month'].min()
        x_max_display = df_window['month'].max()
    else:
        # Fallback to full range if no data in window (e.g., slider covers no data)
        x_min_display = pd.to_datetime(months_for_slider[0])
        x_max_display = pd.to_datetime(months_for_slider[-1])


fig.update_layout(
    xaxis=dict(
        title='Time',
        tickformat=x_axis_tickformat,
        range=[x_min_display, x_max_display] if x_min_display is not None else None,
    ),
    yaxis=dict( # This is the primary 'y' or 'y1' axis for sentiment
        title='Sentiment Consensus (Std Dev)',
        autorange='reversed', # Keep this if lower std dev is better consensus
        side='left',
    ),
    legend=dict(title='Source'),
    height=900,
    margin=dict(l=40, r=150, t=60, b=300),
    plot_bgcolor='white',
    **metric_yaxis_layout_config
)

# Add event line and annotation for the SELECTED event
if selected_event and selected_event != "None":
    event_name_full = selected_event.split(' ', 1)[1]
    event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_full])

    # Check if the event date is within the CURRENTLY DISPLAYED x-axis range
    if x_min_display and x_max_display and x_min_display <= event_date <= x_max_display:
        fig.add_vline(
            x=event_date.to_pydatetime(),
            line_width=2,
            line_dash="dash",
            line_color="red",
        )

        fig.add_annotation(
            x=event_date.to_pydatetime(),
            y=-0.05,
            xref="x",
            yref="paper",
            text=SHORT_EVENT_LABELS.get(event_name_full, event_name_full),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="darkgrey",
            ax=0,
            ay=100,
            textangle=-90,
            valign="bottom",
            font=dict(color="darkgrey", size=10)
        )
# No warning if no event data, because the plot will just be empty for that event.
# The user knows they selected an event, if no data shows, it means no data for that event.


# Animation (This part is still set up for monthly data, as stated in previous comments)
# It will not apply when an event is selected, due to `show_animation` being disabled.
if show_animation:
    frames = []
    # If no event is selected, use df_window's months for animation
    animation_months = sorted(df_window['month'].unique())

    if not animation_months:
        st.warning("No monthly data available for animation in the selected window.")
    else:
        for m_idx, m in enumerate(animation_months):
            frame_updates_for_traces = []

            # Monthly data traces (article and monthly tweet)
            for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
                d_for_frame = df_window[(df_window['source'] == source) & (df_window['month'] <= m)]

                marker_colors = []
                marker_opacities = []
                if not d_for_frame.empty:
                    for i, row in d_for_frame.iterrows():
                        marker_colors.append(sentiment_color(row['mean_sentiment']))
                        if row['month'] < m:
                            marker_opacities.append(0.3)
                        else:
                            marker_opacities.append(1.0)

                frame_updates_for_traces.append({
                    'x': d_for_frame['month'].tolist() if not d_for_frame.empty else [],
                    'y': d_for_frame['std_sentiment'].tolist() if not d_for_frame.empty else [],
                    'marker': {
                        'size': np.sqrt(d_for_frame['count']) * (0.1 if source == 'tweet' else 3) if not d_for_frame.empty else [],
                        'color': marker_colors,
                        'symbol': shape,
                        'line': dict(width=1, color='black'),
                        'opacity': marker_opacities
                    },
                    'text': [
                        f"{source.capitalize()} (Monthly)<br>Month: {mo.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
                        for mo, s, c, cnt in zip(d_for_frame['month'], d_for_frame['mean_sentiment'], d_for_frame['std_sentiment'], d_for_frame['count'])
                    ] if not d_for_frame.empty else [],
                })

            # Metric traces in animation (cumulative)
            for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
                # Adjust index if hourly event traces are present (not in animation case here)
                # In animation, we only have monthly sentiment traces (0, 1) initially
                # So metric traces start from index 2.
                metric_trace_idx = 2 + metric_idx
                d_metric_for_frame = metric_df[metric_df['month'] <= m] # Cumulative data for metric
                frame_updates_for_traces.append({
                    'x': d_metric_for_frame['month'].tolist() if not d_metric_for_frame.empty else [],
                    'y': d_metric_for_frame[metric].tolist() if not d_metric_for_frame.empty else [],
                })

            frames.append(go.Frame(data=frame_updates_for_traces, name=str(m)))
        if frames:
            fig.frames = frames
            fig.update_layout(
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate"
                                }
                            ],
                            "args2": [ # Args when button is pressed again (to stop)
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"
                                }
                            ],
                        }
                    ]
                }]
            )
        # else: st.warning("No data available for animation in the selected window.") # This warning is already handled above

# Update layout - X-axis configuration is now handled in the 'if selected_event' block

st.plotly_chart(fig, use_container_width=True)
