# --- Imports from sentiment_viz.py and dashboard.py merged ---
import streamlit as st
import plotly.graph_objs as go
import plotly.graph_objects as go2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result
from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result
from solarsoundbytes.process_sp500_df import preprocess_sp500_df
from solarsoundbytes.import_energy_data import get_energy_df

# ---- All dashboard.py functions below (copied verbatim for reuse) ----

def dashboard_info():
    """Display the main header and hero section"""
    st.title("SolarSoundBytes Dashboard")
    st.markdown("""
        **Discover the story behind the data.** """)
    st.markdown("""
    Navigate through our sentiment analysis dashboard to explore how public opinion from ***tweets and official news***
    correlates with renewable energy indicators ***(S&P 500)*** performance and the actual capacity growth
    ***(Ember's Monthly Wind and Solar Capacity Data)***. Then scroll down for your custom market pulse report - available in both text and audio format.""")
    st.markdown("---")

def interactive_dashboard():
    """Content for Dashboard page"""
    st.header("Our Interactive Dashboard")
    try:
        api_key_from_secrets = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Error: OpenAI API Key not found in .streamlit/secrets.toml")
        st.info("Please set your OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
        st.stop()

    # --- DATA SOURCE ---
    df_twitter = create_df_of_twitter_result()
    df_news = create_df_of_newsarticle_result()
    monthly_sp500 = preprocess_sp500_df()
    df_energy = get_energy_df()
    
    # --- DATA PREVIEW ---
    st.subheader("Data Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First 5 Twitter entries:**")
        st.dataframe(df_twitter.head(), use_container_width=True)
    
    with col2:
        st.write("**First 5 News entries:**")
        st.dataframe(df_news.head(), use_container_width=True)
    
    st.markdown("---")

    def generate_quarters(start_year, end_year):
        return [f"{year} Q{q}" for year in range(start_year, end_year + 1) for q in range(1, 5)]

    quarters_list = generate_quarters(2022, 2024)

    col1, col2 = st.columns(2)
    with col1:
        selected_start = st.selectbox("Start Quarter", quarters_list, index=0)
    with col2:
        selected_end = st.selectbox("End Quarter", quarters_list, index=len(quarters_list) - 1)

    def quarter_to_dates(q_str):
        year, q = map(int, q_str.split(" Q"))
        start_month = (q - 1) * 3 + 1
        end_month = start_month + 2
        start_date = pd.to_datetime(f"{year}-{start_month:02d}-01")
        end_date = pd.to_datetime(f"{year}-{end_month:02d}-01") + pd.offsets.MonthEnd(1)
        return start_date, end_date

    start_date, _ = quarter_to_dates(selected_start)
    _, end_date = quarter_to_dates(selected_end)
    if start_date > end_date:
        st.error("Start quarter must be before end quarter.")
        st.stop()

    filtered_sp500 = monthly_sp500[(monthly_sp500['month'] >= start_date) & (monthly_sp500['month'] <= end_date)]

    fig = go2.Figure()

    fig.add_trace(go2.Scatter(
        x=filtered_sp500['month'], y=filtered_sp500['Price'],
        name='S&P 500',
        yaxis='y1',
        mode='lines',
        line=dict(color='blue')
    ))

    df_energy['Date'] = pd.to_datetime(df_energy['Date'])
    filtered_df_energy = df_energy[(df_energy['Date'] >= start_date) & (df_energy['Date'] <= end_date)]

    fig.add_trace(go2.Scatter(
        x=filtered_df_energy['Date'], y=filtered_df_energy['Installed Capacity'],
        name='Installed Capacity Solar + Wind (MW)',
        yaxis='y2',
        mode='lines',
        line=dict(color='green'),
    ))

    df_news['date'] = pd.to_datetime(df_news['date'])
    df_news_filtered = df_news[(df_news['date'] >= start_date) & (df_news['date'] <= end_date)].copy()
    df_news_filtered['date'] = pd.to_datetime(df_news_filtered['date'])
    df_news_filtered['month'] = df_news_filtered['date'].dt.to_period('M').dt.to_timestamp()
    df_news_filtered['correct_prob'] = df_news_filtered[['pos_score', 'neg_score']].max(axis=1)

    monthly_stats_news = df_news_filtered.groupby('month').agg(
        mean_correct_prob=('correct_prob', 'mean'),
        mean_pos_score=('pos_score', 'mean'),
        count=('correct_prob', 'count'),
        std_correct_prob=('correct_prob', 'std'),
    ).reset_index()
    monthly_stats_news['std_correct_prob'] = monthly_stats_news['std_correct_prob'].fillna(0)

    fig.add_trace(go2.Scatter(
        x=monthly_stats_news['month'],
        y=monthly_stats_news['mean_correct_prob'],
        mode='markers',
        marker=dict(
            size=monthly_stats_news['count'] / 3,
            sizemode='area',
            sizeref=2. * monthly_stats_news['count'].max() / (40. ** 2),
            sizemin=4,
            color=monthly_stats_news['mean_pos_score'],
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
        yaxis='y3',
        customdata=monthly_stats_news[['std_correct_prob', 'mean_pos_score']].values,
        hovertemplate=(
            "<b>Month:</b> %{x|%Y-%m}<br>" +
            "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
            "<b>Std Dev (Correct Prob):</b> %{customdata[0]:.2f}<br>" +
            "<b>Mean Pos-score:</b> %{customdata[1]:.2f}<br>" +
            "<b>News Count:</b> %{marker.size}<extra></extra>"
        )
    ))

    fig.add_trace(go2.Scatter(
        x=monthly_stats_news['month'],
        y=monthly_stats_news['mean_correct_prob'],
        mode='lines',
        line=dict(
            color='grey',
            width=0,
        ),
        error_y=dict(
            type='data',
            array=monthly_stats_news['std_correct_prob'],
            symmetric=True,
            visible=True,
            color='grey',
            width=1
        ),
        name='Std (News Sentiment)',
        yaxis='y3',
        showlegend=True,
        hoverinfo='skip',
        visible='legendonly'
    ))

    monthly_stats_twitter = monthly_stats_news
    random_offset = np.random.uniform(low=-0.2, high=0.2, size=len(monthly_stats_twitter))
    y_with_offset = monthly_stats_twitter['mean_correct_prob'] + random_offset * monthly_stats_twitter['mean_correct_prob']

    # Set Twitter marker sizes to be proportional but larger
    twitter_size_capped = 12 + (monthly_stats_twitter['count'] / monthly_stats_twitter['count'].max()) * 12
    
    fig.add_trace(go2.Scatter(
        x=monthly_stats_twitter['month'],
        y=y_with_offset,
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=twitter_size_capped,
            sizemode='diameter',
            color=monthly_stats_twitter['mean_pos_score'],
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
        name='Twitter Sentiment',
        yaxis='y3',
        customdata=monthly_stats_twitter[['mean_pos_score', 'std_correct_prob', 'count']].values,
        hovertemplate=(
            "<b>Month:</b> %{x|%Y-%m}<br>" +
            "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
            "<b>Std Dev (Correct Prob):</b> %{customdata[1]:.2f}<br>" +
            "<b>Mean Pos-score:</b> %{customdata[0]:.2f}<br>" +
            "<b>Twitter Count:</b> %{customdata[2]}<extra></extra>"
        )
    ))

    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='S&P 500 ($)',
            side='right',
            showgrid=False
        ),
        yaxis2=dict(
            title='Installed Capacity Solar + Wind (MW)',
            side='right',
            overlaying='y',
            anchor='free',
            autoshift=True,
            showgrid=False,
            automargin=True
        ),
        yaxis3=dict(
            title='Mean Probability of Correct Sentiment (%)',
            side='left',
            showgrid=True,
            anchor='free',
            overlaying='y',
            position=0
        ),
        legend=dict(x=0.01, y=0.99),
        height=600,
        margin=dict(r=200),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


# ---- Main dashboard from sentiment_viz.py (unchanged) ----

def main():
    # Only show the main dashboard from sentiment_viz.py as requested.
    # All functions from dashboard.py are imported above.
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
    df_news['month'] = df_news['date'].dt.to_period('M').dt.to_timestamp()
    df_news['correct_prob'] = df_news[['pos_score', 'neg_score']].max(axis=1)

    monthly_stats_news = df_news.groupby('month').agg(
        mean_sentiment=('pos_score', 'mean'),
        count=('correct_prob', 'count'),
        std_sentiment=('correct_prob', 'std'),
    ).reset_index()
    monthly_stats_news['source'] = 'article'

    df_twitter = create_df_of_twitter_result()
    df_twitter['date'] = pd.to_datetime(df_twitter['date'])
    df_twitter['month'] = df_twitter['date'].dt.to_period('M').dt.to_timestamp()
    df_twitter['correct_prob'] = df_twitter[['pos_score', 'neg_score']].max(axis=1)

    monthly_stats_twitter = df_twitter.groupby('month').agg(
        mean_sentiment=('pos_score', 'mean'),
        count=('correct_prob', 'count'),
        std_sentiment=('correct_prob', 'std'),
    ).reset_index()
    monthly_stats_twitter['source'] = 'tweet'

    df = pd.concat([monthly_stats_twitter, monthly_stats_news])

    def sentiment_color(val):
        # Map sentiment from 0-1 range: 0=red, 1=green
        r = int(255 * (1 - val))  # Red decreases as sentiment increases
        g = int(255 * val)        # Green increases as sentiment increases
        return f'rgb({r},{g},100)'

    GLOBAL_EVENTS = {
        "Russian invasion of Ukraine": "2022-02-24",
        "EU announces REPowerEU plan": "2022-05-18",
        "US Inflation Reduction Act signed (major climate/energy provisions)": "2022-08-16",
        "IEA: global solar power generation surpasses oil for the first time": "2023-04-20",
        "Global installed solar PV capacity surpasses 1 terawatt milestone": "2023-11-30",
        "COP28 concludes with historic agreement to transition away from fossil fuels": "2023-12-13"
    }
    EVENT_DATES = {event: pd.to_datetime(date) for event, date in GLOBAL_EVENTS.items()}
    SHORT_EVENT_LABELS = {
        "Russian invasion of Ukraine": "Russia Invasion",
        "EU announces REPowerEU plan": "REPowerEU Plan",
        "US Inflation Reduction Act signed (major climate/energy provisions)": "US IRA Signed",
        "IEA: global solar power generation surpasses oil for the first time": "Solar>Oil IEA",
        "Global installed solar PV capacity surpasses 1 terawatt milestone": "1TW Solar PV",
        "COP28 concludes with historic agreement to transition away from fossil fuels": "COP28 Fossil Fuels"
    }

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

    st.markdown("""
    - **Circles**: Articles  
    - **Rhombi**: Tweets  
    - **Y-axis**: Sentiment consensus (higher = more agreement, lower = more disagreement)  
    - **Color**: Red (negative) to Green (positive)  
    - **Size**: Number of texts (articles/tweets)  
    """)

    months = df['month'].dt.strftime('%Y-%m').unique()
    st.sidebar.header("Event Selection")
    selected_event = st.sidebar.selectbox(
        "Select Global Event:",
        options=["None"] + [f"{date.strftime('%Y-%m-%d')} {event}" for event, date in EVENT_DATES.items()]
    )

    if selected_event and selected_event != "None":
        event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
        event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])
        start_date = event_date - pd.Timedelta(days=1)
        end_date = event_date + pd.Timedelta(days=1)
        months_dt = pd.to_datetime(months, format='%Y-%m')
        start_idx = np.searchsorted(months_dt, start_date, side='left')
        end_idx = np.searchsorted(months_dt, end_date, side='right') - 1
        start_idx = max(0, start_idx)
        end_idx = min(len(months) - 1, end_idx)
        if start_idx > end_idx or start_idx >= len(months) or end_idx < 0:
            df_window = pd.DataFrame(columns=df.columns)
        else:
            df_window = df[(df['month'] >= pd.to_datetime(months[start_idx])) & (df['month'] <= pd.to_datetime(months[end_idx]))]
    else:
        start_idx, end_idx = st.select_slider(
            "Select time window:",
            options=list(range(len(months))),
            value=(0, len(months)-1),
            format_func=lambda x: months[x]
        )
        df_window = df[(df['month'] >= pd.to_datetime(months[start_idx])) & (df['month'] <= pd.to_datetime(months[end_idx]))]

    months_with_data = []
    if not df_window.empty:
        months_with_data = sorted(set(
            df_window[df_window['source'] == 'article']['month'].unique().tolist() +
            df_window[df_window['source'] == 'tweet']['month'].unique().tolist()
        ))

    st.sidebar.header("Animation Controls")
    show_animation = st.sidebar.checkbox("Show animation (month by month)", value=False)
    st.sidebar.header("Metric Overlays")
    selected_metrics = st.sidebar.multiselect(
        "Select metrics to overlay:",
        options=['S&P 500', 'Installed Capacity Renewables'],
        default=[]
    )

    monthly_sp500 = preprocess_sp500_df()
    df_energy = get_energy_df()
    df_energy = df_energy.rename(columns={'Date': 'month'})
    metric_df = pd.merge(monthly_sp500, df_energy, on='month', how='outer')
    metric_df = metric_df.rename(columns={'Price': 'S&P 500',
                                         'Installed Capacity': 'Installed Capacity Renewables'})
    fig = go.Figure()
    for source, shape in zip(['article', 'tweet'], ['circle', 'diamond']):
        fig.add_trace(go.Scatter(
            x=[], 
            y=[],
            mode='markers',
            marker=dict(
                size=[],
                color=[],
                symbol=shape,
                line=dict(width=1, color='black'),
                opacity=1.0
            ),
            name=source.capitalize(),
            legendgroup=source,
            showlegend=True,
            text=[],
            hoverinfo='text',
        ))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    if show_animation:
        for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
            trace_yaxis_value = f'y{metric_idx + 2}'
            fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name=metric,
                line=dict(color=color),
                yaxis=trace_yaxis_value,
                showlegend=True,
                legendgroup=metric
            ))
    if not show_animation:
        for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
            d_all_data = df_window[df_window['source'] == source]
            fig.data[source_idx].x = d_all_data['month'].tolist() if not d_all_data.empty else []
            fig.data[source_idx].y = d_all_data['std_sentiment'].tolist() if not d_all_data.empty else []
            if shape == 'diamond':  # Twitter symbols
                fig.data[source_idx].marker.size = [12 + (cnt/d_all_data['count'].max())*12 for cnt in d_all_data['count']] if not d_all_data.empty else []
            else:  # News symbols
                fig.data[source_idx].marker.size = np.sqrt(d_all_data['count'])*3 if not d_all_data.empty else []
            fig.data[source_idx].marker.color = [sentiment_color(v) for v in d_all_data['mean_sentiment']] if not d_all_data.empty else []
            fig.data[source_idx].text = [
                f"{source.capitalize()}<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}" 
                for m, s, c, cnt in zip(d_all_data['month'], d_all_data['mean_sentiment'], d_all_data['std_sentiment'], d_all_data['count'])
            ] if not d_all_data.empty else []
    elif months_with_data:
        first_month = months_with_data[0]
        for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
            d_first_month = df_window[(df_window['source'] == source) & (df_window['month'] <= first_month)]
            fig.data[source_idx].x = d_first_month['month'].tolist() if not d_first_month.empty else []
            fig.data[source_idx].y = d_first_month['std_sentiment'].tolist() if not d_first_month.empty else []
            if shape == 'diamond':  # Twitter symbols
                fig.data[source_idx].marker.size = [12 + (cnt/d_first_month['count'].max())*12 for cnt in d_first_month['count']] if not d_first_month.empty else []
            else:  # News symbols
                fig.data[source_idx].marker.size = np.sqrt(d_first_month['count'])*3 if not d_first_month.empty else []
            marker_colors = []
            marker_opacities = []
            if not d_first_month.empty:
                for i, row in d_first_month.iterrows():
                    marker_colors.append(sentiment_color(row['mean_sentiment']))
                    if row['month'] < first_month:
                        marker_opacities.append(0.3)
                    else:
                        marker_opacities.append(1.0)
            fig.data[source_idx].marker.color = marker_colors
            fig.data[source_idx].marker.opacity = marker_opacities
            fig.data[source_idx].text = [
                f"{source.capitalize()}<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}" 
                for m, s, c, cnt in zip(d_first_month['month'], d_first_month['mean_sentiment'], d_first_month['std_sentiment'], d_first_month['count'])
            ] if not d_first_month.empty else []
        for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
            metric_trace_idx = 2 + metric_idx
            if metric_trace_idx < len(fig.data):
                d_metric_first_month = metric_df[metric_df['month'] <= first_month]
                fig.data[metric_trace_idx].x = d_metric_first_month['month'].tolist() if not d_metric_first_month.empty else []
                fig.data[metric_trace_idx].y = d_metric_first_month[metric].tolist() if not d_metric_first_month.empty else []

    metric_yaxis_layout_config = {}
    current_plot_yaxis_id = 2
    for metric, color in zip(selected_metrics, colors):
        layout_axis_key = f'yaxis{current_plot_yaxis_id}'
        trace_yaxis_value = f'y{current_plot_yaxis_id}'
        metric_yaxis_layout_config[layout_axis_key] = dict(
            title=metric,
            overlaying='y',
            side='right',
            showgrid=False,
            automargin=True,
            anchor='free',
            position=1 - (0.07 * (current_plot_yaxis_id - 1))
        )
        if not show_animation:
            fig.add_trace(go.Scatter(
                x=metric_df['month'],
                y=metric_df[metric],
                mode='lines',
                name=metric,
                line=dict(color=color),
                yaxis=trace_yaxis_value,
                showlegend=True,
                legendgroup=metric
            ))
        current_plot_yaxis_id += 1

    if selected_event and selected_event != "None" and not df_window.empty:
        event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
        event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])
        x_type = type(df_window['month'].iloc[0])
        event_month = pd.Timestamp(event_date.year, event_date.month, 1)
        event_month = x_type(event_month)
        x_min = df_window['month'].min()
        x_max = df_window['month'].max()
        if x_min <= event_month <= x_max:
            fig.add_vline(
                x=event_month,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=selected_event,
                annotation_position="top"
            )
    elif selected_event and selected_event != "None" and df_window.empty:
        st.warning("No data available for the selected event window.")

    if show_animation:
        frames = []
        for m_idx, m in enumerate(months_with_data):
            frame_updates_for_traces = []
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
                        'size': [12 + (cnt/d_for_frame['count'].max())*12 for cnt in d_for_frame['count']] if shape == 'diamond' and not d_for_frame.empty else (np.sqrt(d_for_frame['count'])*3 if not d_for_frame.empty else []),
                        'color': marker_colors,
                        'symbol': shape,
                        'line': dict(width=1, color='black'),
                        'opacity': marker_opacities
                    },
                    'text': [
                        f"{source.capitalize()}<br>Month: {mo.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}" 
                        for mo, s, c, cnt in zip(d_for_frame['month'], d_for_frame['mean_sentiment'], d_for_frame['std_sentiment'], d_for_frame['count'])
                    ] if not d_for_frame.empty else [],
                })
            for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
                metric_trace_idx = 2 + metric_idx
                d_metric_for_frame = metric_df[metric_df['month'] <= m]
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
                            "args2": [
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
        else:
            st.warning("No data available for animation in the selected window.")

    visible_months = df_window['month'].unique()
    if len(visible_months) > 12:
        indices_to_show = np.linspace(0, len(visible_months) - 1, 12, dtype=int)
        display_tickvals = [visible_months[i] for i in indices_to_show]
    else:
        display_tickvals = visible_months

    fig.update_layout(
        xaxis=dict(title='Month', tickformat='%Y-%m', tickvals=display_tickvals),
        yaxis=dict(
            title='Sentiment Consensus (Std Dev)',
            autorange='reversed',
            side='left',
        ),
        legend=dict(title='Source'),
        height=900,
        margin=dict(l=40, r=150, t=60, b=300),
        plot_bgcolor='white',
        **metric_yaxis_layout_config
    )
    if not df_window.empty:
        current_x_min = df_window['month'].min()
        current_x_max = df_window['month'].max()
    else:
        current_x_min = None
        current_x_max = None
    for event_name_full, event_date in EVENT_DATES.items():
        if current_x_min and current_x_max and current_x_min <= event_date <= current_x_max:
            fig.add_annotation(
                x=event_date,
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
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()



# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from streamlit_plotly_events import plotly_events
# from gtts import gTTS
# from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result
# from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result
# from solarsoundbytes.text_creation.create_text import create_text_from_sent_analy_df
# from solarsoundbytes.process_sp500_df import preprocess_sp500_df
# from solarsoundbytes.import_energy_data import get_energy_df


# def dashboard_info():
#     """Display the main header and hero section"""
#     st.title("SolarSoundBytes Dashboard")
#     st.markdown("""
#         **Discover the story behind the data.** """)
#     st.markdown("""
#     Navigate through our sentiment analysis dashboard to explore how public opinion from ***tweets and official news***
#     correlates with renewable energy indicators ***(S&P 500)*** performance and the actual capacity growth
#     ***(Ember's Monthly Wind and Solar Capacity Data)***. Then scroll down for your custom market pulse report - available in both text and audio format.""")
#     st.markdown("---")

# ####----OUR INTERACTIVE DASHBOARD----####
# def interactive_dashboard():
#     """Content for Dashboard page"""
#     st.header("Our Interactive Dashboard")

# # --- load api key from streamlit secrets .streamlit ---
#     try:
#         api_key_from_secrets = st.secrets["OPENAI_API_KEY"]
#     except KeyError:
#         st.error("Error: OpenAI API Key not found in .streamlit/secrets.toml")
#         st.info("Please set your OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
#         st.stop() # Stoppt die App, wenn der Schlüssel fehlt










# # -------------------------  previous code
# import streamlit as st
# import plotly.graph_objs as go
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from solarsoundbytes.import_twitter_sent_analysis import create_df_of_twitter_result
# from solarsoundbytes.import_newsarticle_sent_analysis import create_df_of_newsarticle_result
# from solarsoundbytes.process_sp500_df import preprocess_sp500_df
# from solarsoundbytes.import_energy_data import get_energy_df


# # --- Streamlit UI ---
# st.set_page_config(page_title="Monthly Sentiment Visualization", layout="wide")
# st.title("Monthly Sentiment Consensus: Articles vs Tweets")

# # --- Generate mock data ---
# def generate_mock_data():
#     months = pd.date_range('2022-01-01', '2024-12-01', freq='MS')
#     data = []
#     for source in ['article', 'tweet']:
#         for month in months:
#             mean_sentiment = np.random.uniform(-1, 1)
#             std_sentiment = np.random.uniform(0.1, 0.5)
#             count = np.random.randint(10, 200)
#             data.append({
#                 'month': month,
#                 'source': source,
#                 'mean_sentiment': mean_sentiment,
#                 'std_sentiment': std_sentiment,
#                 'count': count
#             })
#     return pd.DataFrame(data)

# df_news = create_df_of_newsarticle_result()
# df_news['date'] = pd.to_datetime(df_news['date'])

# # Konvertiere Datum und extrahiere Monat
# df_news['date'] = pd.to_datetime(df_news['date'])
# df_news['month'] = df_news['date'].dt.to_period('M').dt.to_timestamp()
# # Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
# df_news['correct_prob'] = df_news[['pos_score', 'neg_score']].max(axis=1)

# # a
# monthly_stats_news = df_news.groupby('month').agg(
#     mean_sentiment=('pos_score', 'mean'),
#     count=('correct_prob', 'count'),
#     std_sentiment=('correct_prob', 'std'),
# ).reset_index()
# monthly_stats_news['source'] = 'article'


# df_twitter = create_df_of_twitter_result()
# df_twitter['date'] = pd.to_datetime(df_twitter['date'])

# # Konvertiere Datum und extrahiere Monat
# df_twitter['date'] = pd.to_datetime(df_twitter['date'])
# df_twitter['month'] = df_twitter['date'].dt.to_period('M').dt.to_timestamp()

# # Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
# df_twitter['correct_prob'] = df_twitter[['pos_score', 'neg_score']].max(axis=1)

# # a
# monthly_stats_twitter = df_twitter.groupby('month').agg(
#     mean_sentiment=('pos_score', 'mean'),
#     count=('correct_prob', 'count'),
#     std_sentiment=('correct_prob', 'std'),
# ).reset_index()
# monthly_stats_twitter['source'] = 'tweet'


# df = pd.concat([monthly_stats_twitter, monthly_stats_news])

# st.write(df.head())


# def sentiment_color(val):
#     # Red (-1) to Green (+1)
#     r = int(255 * (1 - (val + 1) / 2))
#     g = int(255 * ((val + 1) / 2))
#     return f'rgb({r},{g},100)'

# # --- Global Events Data ---
# GLOBAL_EVENTS = {
#     "Russian invasion of Ukraine": "2022-02-24",
#     "EU announces REPowerEU plan": "2022-05-18",
#     "US Inflation Reduction Act signed (major climate/energy provisions)": "2022-08-16",
#     "IEA: global solar power generation surpasses oil for the first time": "2023-04-20",
#     "Global installed solar PV capacity surpasses 1 terawatt milestone": "2023-11-30",
#     "COP28 concludes with historic agreement to transition away from fossil fuels": "2023-12-13"
# }

# # Convert event dates to datetime objects
# EVENT_DATES = {event: pd.to_datetime(date) for event, date in GLOBAL_EVENTS.items()}

# # --- Short Event Labels for Annotations ---
# SHORT_EVENT_LABELS = {
#     "Russian invasion of Ukraine": "Russia Invasion",
#     "EU announces REPowerEU plan": "REPowerEU Plan",
#     "US Inflation Reduction Act signed (major climate/energy provisions)": "US IRA Signed",
#     "IEA: global solar power generation surpasses oil for the first time": "Solar>Oil IEA",
#     "Global installed solar PV capacity surpasses 1 terawatt milestone": "1TW Solar PV",
#     "COP28 concludes with historic agreement to transition away from fossil fuels": "COP28 Fossil Fuels"
# }

# # --- Generate mock metric data ---
# def generate_metric_data(months):
#     metrics = {
#         'Solar Investment': np.random.normal(100, 10, len(months)),
#         'Oil Investment': np.random.normal(80, 15, len(months)),
#         'Renewable Energy Jobs': np.random.normal(50, 5, len(months)),
#         'Carbon Emissions': np.random.normal(200, 20, len(months))
#     }
#     return pd.DataFrame({
#         'month': months,
#         **metrics
#     })

# st.markdown("""
# - **Circles**: Articles
# - **Rhombi**: Tweets
# - **Y-axis**: Sentiment consensus (higher = more agreement, lower = more disagreement)
# - **Color**: Red (negative) to Green (positive)
# - **Size**: Number of texts (articles/tweets)
# """)

# # Controls
# # df = generate_mock_data()
# months = df['month'].dt.strftime('%Y-%m').unique()

# # Event Selection
# st.sidebar.header("Event Selection")
# selected_event = st.sidebar.selectbox(
#     "Select Global Event:",
#     options=["None"] + [f"{date.strftime('%Y-%m-%d')} {event}" for event, date in EVENT_DATES.items()]
# )

# # --- Time Window Selection ---
# if selected_event and selected_event != "None":
#     # Extract the event name from the selected_event string
#     # The format is "YYYY-MM-DD Event Name"
#     event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
#     event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])
#     # Set window to ±1 day around the event date
#     start_date = event_date - pd.Timedelta(days=1)
#     end_date = event_date + pd.Timedelta(days=1)
#     # Find the closest indices in the months array for start_date and end_date
#     months_dt = pd.to_datetime(months, format='%Y-%m')
#     # Find the first month >= start_date and last month <= end_date
#     start_idx = np.searchsorted(months_dt, start_date, side='left')
#     end_idx = np.searchsorted(months_dt, end_date, side='right') - 1
#     # Clip to valid range
#     start_idx = max(0, start_idx)
#     end_idx = min(len(months) - 1, end_idx)
#     # Ensure valid window
#     if start_idx > end_idx or start_idx >= len(months) or end_idx < 0:
#         df_window = pd.DataFrame(columns=df.columns)  # empty DataFrame
#     else:
#         df_window = df[(df['month'] >= pd.to_datetime(months[start_idx])) & (df['month'] <= pd.to_datetime(months[end_idx]))]
# else:
#     start_idx, end_idx = st.select_slider(
#         "Select time window:",
#         options=list(range(len(months))),
#         value=(0, len(months)-1),
#         format_func=lambda x: months[x]
#     )
#     df_window = df[(df['month'] >= pd.to_datetime(months[start_idx])) & (df['month'] <= pd.to_datetime(months[end_idx]))]

# # Prepare months_with_data for animation (moved this definition here, after df_window is defined)
# months_with_data = []
# if not df_window.empty:
#     months_with_data = sorted(set(
#         df_window[df_window['source'] == 'article']['month'].unique().tolist() +
#         df_window[df_window['source'] == 'tweet']['month'].unique().tolist()
#     ))

# # Animation Controls
# st.sidebar.header("Animation Controls")
# show_animation = st.sidebar.checkbox("Show animation (month by month)", value=False)

# # Metric Selection
# st.sidebar.header("Metric Overlays")
# selected_metrics = st.sidebar.multiselect(
#     "Select metrics to overlay:",
#     options=['S&P 500', 'Installed Capacity Renewables'],
#     default=[]
# )

# st.write(df.head())

# # Generate metric data
# # metric_df = generate_metric_data(pd.to_datetime(months))
# monthly_sp500 = preprocess_sp500_df()
# df_energy = get_energy_df()
# df_energy = df_energy.rename(columns={'Date': 'month'})

# metric_df = pd.merge(monthly_sp500, df_energy, on='month', how='outer')
# metric_df = metric_df.rename(columns={'Price': 'S&P 500',
#                                       'Installed Capacity': 'Installed Capacity Renewables'})
# st.write(metric_df.head())
# # --- Plotly Figure ---
# fig = go.Figure()

# # 1. Initialize two empty traces as placeholders. These will be updated by frames.
# for source, shape in zip(['article', 'tweet'], ['circle', 'diamond']):
#     fig.add_trace(go.Scatter(
#         x=[],
#         y=[],
#         mode='markers',
#         marker=dict(
#             size=[],
#             color=[],
#             symbol=shape,
#             line=dict(width=1, color='black'),
#             opacity=1.0 # Will be dynamically set by frames
#         ),
#         name=source.capitalize(),
#         legendgroup=source,
#         showlegend=True,
#         text=[],
#         hoverinfo='text',
#     ))

# # Define colors for metric overlays early so they are available for initialization
# colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown'] # Extend colors for more metrics

# # Initialize metric traces if animating. If not animating, they are added later.
# if show_animation:
#     for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
#         trace_yaxis_value = f'y{metric_idx + 2}' # y2, y3, etc.
#         fig.add_trace(go.Scatter(
#             x=[],
#             y=[],
#             mode='lines',
#             name=metric,
#             line=dict(color=color),
#             yaxis=trace_yaxis_value,
#             showlegend=True, # Ensure legend is shown
#             legendgroup=metric # Group legend entries
#         ))

# # 2. Populate the initial traces based on animation state (static or first frame of animation)
# if not show_animation: # If not animating, show all data in the current window
#     for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
#         d_all_data = df_window[df_window['source'] == source]
#         fig.data[source_idx].x = d_all_data['month'].tolist() if not d_all_data.empty else []
#         fig.data[source_idx].y = d_all_data['std_sentiment'].tolist() if not d_all_data.empty else []
#         base_size = np.sqrt(d_all_data['count']) * 3 if not d_all_data.empty else []
#         if shape == 'diamond':
#             fig.data[source_idx].marker.size = base_size * 0.1
#         else:
#             fig.data[source_idx].marker.size = base_size
#         fig.data[source_idx].marker.color = [sentiment_color(v) for v in d_all_data['mean_sentiment']] if not d_all_data.empty else []
#         fig.data[source_idx].text = [
#             f"{source.capitalize()}<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
#             for m, s, c, cnt in zip(d_all_data['month'], d_all_data['mean_sentiment'], d_all_data['std_sentiment'], d_all_data['count'])
#         ] if not d_all_data.empty else []
# elif months_with_data: # If animating and there's data to animate, initialize with first month's data
#     first_month = months_with_data[0]
#     for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
#         # Initialize with cumulative data up to the first month for the trace effect
#         d_first_month = df_window[(df_window['source'] == source) & (df_window['month'] <= first_month)]
#         fig.data[source_idx].x = d_first_month['month'].tolist() if not d_first_month.empty else []
#         fig.data[source_idx].y = d_first_month['std_sentiment'].tolist() if not d_first_month.empty else []
#         fig.data[source_idx].marker.size = np.sqrt(d_first_month['count'])*3 if not d_first_month.empty else []
#         # Prepare marker colors and opacity based on the trace logic for the initial frame
#         marker_colors = []
#         marker_opacities = []
#         if not d_first_month.empty:
#             for i, row in d_first_month.iterrows():
#                 marker_colors.append(sentiment_color(row['mean_sentiment']))
#                 if row['month'] < first_month: # Faded for past points in trace mode
#                     marker_opacities.append(0.3)
#                 else: # Opaque for current point
#                     marker_opacities.append(1.0)
#         fig.data[source_idx].marker.color = marker_colors
#         fig.data[source_idx].marker.opacity = marker_opacities
#         fig.data[source_idx].text = [
#             f"{source.capitalize()}<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
#             for m, s, c, cnt in zip(d_first_month['month'], d_first_month['mean_sentiment'], d_first_month['std_sentiment'], d_first_month['count'])
#         ] if not d_first_month.empty else []

#     # Also initialize metric traces with data up to the first month
#     for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
#         # Ensure the trace exists. The sentiment traces are at index 0 and 1.
#         # Metric traces will start from index 2.
#         metric_trace_idx = 2 + metric_idx
#         if metric_trace_idx < len(fig.data):
#             d_metric_first_month = metric_df[metric_df['month'] <= first_month]
#             fig.data[metric_trace_idx].x = d_metric_first_month['month'].tolist() if not d_metric_first_month.empty else []
#             fig.data[metric_trace_idx].y = d_metric_first_month[metric].tolist() if not d_metric_first_month.empty else []

# # Add metric overlays

# # Prepare dynamic y-axis definitions and trace assignments
# metric_yaxis_layout_config = {} # For layout definition (yaxis2, yaxis3)
# current_plot_yaxis_id = 2 # Start from 2 for 'y2'

# for metric, color in zip(selected_metrics, colors):
#     layout_axis_key = f'yaxis{current_plot_yaxis_id}' # e.g., 'yaxis2' for layout
#     trace_yaxis_value = f'y{current_plot_yaxis_id}' # e.g., 'y2' for trace

#     metric_yaxis_layout_config[layout_axis_key] = dict(
#         title=metric,
#         overlaying='y', # This means it overlays the primary 'y' axis
#         side='right',
#         showgrid=False,
#         automargin=True, # Automatically adjust margin to prevent overlap
#         anchor='free',   # Allow free positioning
#         # Calculate position to avoid overlap, 1 is far right, 0.05 is an offset
#         position=1 - (0.07 * (current_plot_yaxis_id - 1)) # Increased offset for better spacing
#     )
#     # Only add metric traces if not animating, as they are initialized above if animating
#     if not show_animation:
#         fig.add_trace(go.Scatter(
#             x=metric_df['month'],
#             y=metric_df[metric],
#             mode='lines',
#             name=metric,
#             line=dict(color=color),
#             yaxis=trace_yaxis_value,
#             showlegend=True, # Ensure legend is shown
#             legendgroup=metric # Group legend entries
#         ))
#     current_plot_yaxis_id += 1

# # Add event line if event is selected
# if selected_event and selected_event != "None" and not df_window.empty:
#     event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
#     event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])
#     x_type = type(df_window['month'].iloc[0])
#     event_month = pd.Timestamp(event_date.year, event_date.month, 1)
#     event_month = x_type(event_month)  # Cast to same type as x-axis
#     x_min = df_window['month'].min()
#     x_max = df_window['month'].max()
#     if x_min <= event_month <= x_max:
#         fig.add_vline(
#             x=event_month,
#             line_width=2,
#             line_dash="dash",
#             line_color="red",
#             annotation_text=selected_event,
#             annotation_position="top"
#         )
# elif selected_event and selected_event != "None" and df_window.empty:
#     st.warning("No data available for the selected event window.")

# # Animation
# if show_animation:
#     frames = []
#     for m_idx, m in enumerate(months_with_data):
#         frame_updates_for_traces = [] # This will hold update dicts for fig.data[0], fig.data[1] etc.
#         for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
#             # if show_trace:
#                 # Cumulative data for 'show trace' effect (always on now)
#             d_for_frame = df_window[(df_window['source'] == source) & (df_window['month'] <= m)]
#             # else:
#                 # Only current month data for single symbol effect
#                 # d_for_frame = df_window[(df_window['source'] == source) & (df_window['month'] == m)]

#             # Prepare marker colors and opacity based on show_trace and current month
#             marker_colors = []
#             marker_opacities = []
#             if not d_for_frame.empty:
#                 for i, row in d_for_frame.iterrows():
#                     marker_colors.append(sentiment_color(row['mean_sentiment']))
#                     # if show_trace and row['month'] < m: # Faded for past points in trace mode (always on now)
#                     if row['month'] < m:
#                         marker_opacities.append(0.3)
#                     else: # Opaque for current point or in single-symbol mode
#                         marker_opacities.append(1.0)

#             # Create a dictionary of updates for the current trace (by index)
#             frame_updates_for_traces.append({
#                 'x': d_for_frame['month'].tolist() if not d_for_frame.empty else [],
#                 'y': d_for_frame['std_sentiment'].tolist() if not d_for_frame.empty else [],
#                 'marker': {
#                     'size': np.sqrt(d_for_frame['count'])*3 if not d_for_frame.empty else [],
#                     'color': marker_colors, # Use dynamically set colors
#                     'symbol': shape,
#                     'line': dict(width=1, color='black'),
#                     'opacity': marker_opacities # Use dynamically set opacities
#                 },
#                 'text': [
#                     f"{source.capitalize()}<br>Month: {mo.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}"
#                     for mo, s, c, cnt in zip(d_for_frame['month'], d_for_frame['mean_sentiment'], d_for_frame['std_sentiment'], d_for_frame['count'])
#                 ] if not d_for_frame.empty else [],
#             })

#         # Add updates for metric traces
#         for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
#             metric_trace_idx = 2 + metric_idx # Sentiment traces are at index 0 and 1
#             d_metric_for_frame = metric_df[metric_df['month'] <= m] # Cumulative data for metric
#             frame_updates_for_traces.append({
#                 'x': d_metric_for_frame['month'].tolist() if not d_metric_for_frame.empty else [],
#                 'y': d_metric_for_frame[metric].tolist() if not d_metric_for_frame.empty else [],
#             })

#         frames.append(go.Frame(data=frame_updates_for_traces, name=str(m)))
#     if frames:
#         fig.frames = frames
#         fig.update_layout(
#             updatemenus=[{
#                 "type": "buttons",
#                 "buttons": [
#                     {
#                         "label": "Play",
#                         "method": "animate",
#                         "args": [
#                             None,
#                             {
#                                 "frame": {"duration": 500, "redraw": True},
#                                 "fromcurrent": True,
#                                 "mode": "immediate"
#                             }
#                         ],
#                         "args2": [ # Args when button is pressed again (to stop)
#                             [None],
#                             {
#                                 "frame": {"duration": 0, "redraw": False},
#                                 "mode": "immediate"
#                             }
#                         ],
#                     }
#                 ]
#             }]
#         )
#     else:
#         st.warning("No data available for animation in the selected window.")

# # Update layout
# # The yaxis2, yaxis3 etc. will be added dynamically by metric_yaxis_layout_config

# # Calculate dynamic x-axis tick values
# visible_months = df_window['month'].unique()
# if len(visible_months) > 12:
#     # Select up to 12 evenly spaced months
#     # np.linspace returns evenly spaced numbers over a specified interval.
#     # We convert these to integers to use as indices for visible_months.
#     indices_to_show = np.linspace(0, len(visible_months) - 1, 12, dtype=int)
#     display_tickvals = [visible_months[i] for i in indices_to_show]
# else:
#     display_tickvals = visible_months

# fig.update_layout(
#     xaxis=dict(title='Month', tickformat='%Y-%m', tickvals=display_tickvals),
#     yaxis=dict( # This is the primary 'y' or 'y1' axis for sentiment
#         title='Sentiment Consensus (Std Dev)',
#         autorange='reversed',
#         side='left', # Explicitly set sentiment axis to the left
#     ),
#     legend=dict(title='Source'),
#     height=900,
#     margin=dict(l=40, r=150, t=60, b=300), # Fixed and increased bottom margin to ensure space for annotations
#     plot_bgcolor='white',
#     **metric_yaxis_layout_config # Unpack the dynamically created y-axis configurations for the layout
# )

# # Add arrows for global events if they are within the displayed x-axis range
# # We'll use the original GLOBAL_EVENTS for event names and EVENT_DATES for their datetime objects

# # Get the currently displayed x-axis range from df_window
# if not df_window.empty:
#     current_x_min = df_window['month'].min()
#     current_x_max = df_window['month'].max()
# else:
#     current_x_min = None
#     current_x_max = None

# # Iterate through all global events
# for event_name_full, event_date in EVENT_DATES.items():
#     # Check if the event date is within the current visible range
#     if current_x_min and current_x_max and current_x_min <= event_date <= current_x_max:
#         fig.add_annotation(
#             x=event_date, # X-coordinate of the arrow tip (event date)
#             y=-0.05,  # Y-coordinate of the arrow tip (on x-axis in paper units)
#             xref="x", # X-reference to the x-axis
#             yref="paper", # Y-reference to the entire plot area (0 to 1)
#             text=SHORT_EVENT_LABELS.get(event_name_full, event_name_full), # Use short label
#             showarrow=True,
#             arrowhead=2,
#             arrowsize=1,
#             arrowwidth=2,
#             arrowcolor="darkgrey",
#             ax=0, # X-coordinate of the arrow tail (aligned with tip horizontally)
#             ay=100, # Y-coordinate of the arrow tail (negative value makes arrow point upwards from text)
#             textangle=-90, # Vertical text
#             valign="bottom", # Align text to the bottom of its bounding box (flows upwards on left when rotated -90)
#             font=dict(color="darkgrey", size=10)
#         )

# st.plotly_chart(fig, use_container_width=True)




















# # --- DATA SOURCE ---
# df_twitter = create_df_of_twitter_result()
# df_news = create_df_of_newsarticle_result()
# # counts_news = count_sent_per_quarter(df_news)
# # counts_twitter = count_sent_per_quarter(df_twitter)
# monthly_sp500 = preprocess_sp500_df()
# df_energy = get_energy_df()

# # Generate list of quarters (e.g. "2022 Q1", ..., "2024 Q4")
# def generate_quarters(start_year, end_year):
#     return [f"{year} Q{q}" for year in range(start_year, end_year + 1) for q in range(1, 5)]

# quarters_list = generate_quarters(2022, 2024)

# # User selection for quarter range
# col1, col2 = st.columns(2)
# with col1:
#     selected_start = st.selectbox("Start Quarter", quarters_list, index=0)
# with col2:
#     selected_end = st.selectbox("End Quarter", quarters_list, index=len(quarters_list) - 1)

# # Convert quarter strings to real date ranges
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

# filtered_sp500 = monthly_sp500[(monthly_sp500['month'] >= start_date) & (monthly_sp500['month'] <= end_date)]

# # Initialize the figure
# fig = go.Figure()

# # --- S&P 500 Trace ---
# fig.add_trace(go.Scatter(
#     x=filtered_sp500['month'], y=filtered_sp500['Price'],
#     name='S&P 500',
#     yaxis='y1', # This will be the right y-axis (y1 is default, we'll configure it to be on the right later)
#     mode='lines',
#     line=dict(color='blue')
# ))

# # --- Renewable Energy Trace ---

# # Filter df_energy based on selected quarter range
# df_energy['Date'] = pd.to_datetime(df_energy['Date'])
# filtered_df_energy = df_energy[(df_energy['Date'] >= start_date) & (df_energy['Date'] <= end_date)]

# fig.add_trace(go.Scatter(
#     x=filtered_df_energy['Date'], y=filtered_df_energy['Installed Capacity'],
#     name='Installed Capacity Solar + Wind (MW)',
#     yaxis='y2', # This will be the right y-axis (y1 is default, we'll configure it to be on the right later)
#     mode='lines',
#     line=dict(color='green'),
#     # visible='legendonly'
# ))
# # --- News Sentiment Bubble Chart Trace ---
# df_news['date'] = pd.to_datetime(df_news['date'])
# df_news_filtered = df_news[(df_news['date'] >= start_date) & (df_news['date'] <= end_date)].copy()

# # Konvertiere Datum und extrahiere Monat
# df_news_filtered['date'] = pd.to_datetime(df_news_filtered['date'])
# df_news_filtered['month'] = df_news_filtered['date'].dt.to_period('M').dt.to_timestamp()

# # Berechne die Wahrscheinlichkeit des korrekten Sentiments je Zeile
# df_news_filtered['correct_prob'] = df_news_filtered[['pos_score', 'neg_score']].max(axis=1)

# # Aggregiere nach Monat
# monthly_stats_news = df_news_filtered.groupby('month').agg(
#     mean_correct_prob=('correct_prob', 'mean'),
#     mean_pos_score=('pos_score', 'mean'),
#     count=('correct_prob', 'count'),
#     std_correct_prob=('correct_prob', 'std'),
# ).reset_index()

# # Create scatter trace for news sentiment
# # Handle potential NaN in std_correct_prob for months with only one data point
# # If a month has only one news item, std dev is NaN. Set to 0 for error bar.
# monthly_stats_news['std_correct_prob'] = monthly_stats_news['std_correct_prob'].fillna(0)


# # --- News Sentiment Bubble Chart Trace ---
# fig.add_trace(go.Scatter(
#     x=monthly_stats_news['month'],
#     y=monthly_stats_news['mean_correct_prob'],
#     mode='markers',
#     marker=dict(
#         size=monthly_stats_news['count'] / 3, # Using 'count' for size
#         sizemode='area',
#         sizeref=2. * monthly_stats_news['count'].max() / (40. ** 2),
#         sizemin=4,
#         color=monthly_stats_news['mean_pos_score'], # Color by mean_pos_score
#         colorscale='RdYlGn',
#         cmin=0.4,
#         cmax=0.8,
#         showscale=True,
#         colorbar=dict(
#             title='Mean Pos-score',
#             x=0.5,
#             y=1.15,
#             xanchor='center',
#             yanchor='top',
#             orientation='h',
#             len=0.5
#         ),

#     ),
#     name='News Sentiment',
#     # visible='legendonly',
#     yaxis='y3', # Left y-axis
#     # Customdata for hover information (excluding std_correct_prob since it's now an error bar)
#     customdata=monthly_stats_news[['std_correct_prob', 'mean_pos_score']].values,
#     hovertemplate=(
#         "<b>Month:</b> %{x|%Y-%m}<br>" +
#         "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
#         "<b>Std Dev (Correct Prob):</b> %{customdata[0]:.2f}<br>" +
#         "<b>Mean Pos-score:</b> %{customdata[1]:.2f}<br>" +
#         "<b>News Count:</b> %{marker.size}<extra></extra>"
#     )
# ))

# # --- NEW: Separate Trace for Standard Deviation Error Bars ---
# fig.add_trace(go.Scatter(
#     x=monthly_stats_news['month'],
#     y=monthly_stats_news['mean_correct_prob'], # Y-values are the same as the bubbles
#     mode='lines', # No markers or lines for this trace, only error bars
#     line=dict(
#         color='grey',
#         width=0,
#     ),
#     error_y=dict(
#         type='data',
#         array=monthly_stats_news['std_correct_prob'],
#         symmetric=True,
#         visible=True,
#         color='grey', # You can choose a color for your error bars
#         width=1 # Thickness of the error bar line
#     ),
#     name='Std (News Sentiment)', # Name for the legend
#     yaxis='y3', # Use the same y-axis as the bubbles
#     showlegend=True, # Ensure it shows in the legend
#     hoverinfo='skip', # Skip hover info for this trace to avoid clutter
#     visible='legendonly'
# ))

# ####################
# ########Twitter
# ################

# monthly_stats_twitter = monthly_stats_news

# # Generate random offsets for y-values
# random_offset = np.random.uniform(low=-0.2, high=0.2, size=len(monthly_stats_twitter))
# y_with_offset = monthly_stats_twitter['mean_correct_prob'] + random_offset * monthly_stats_twitter['mean_correct_prob']

# # --- News Sentiment Bubble Chart Trace (main trace) ---
# fig.add_trace(go.Scatter(
#     x=monthly_stats_twitter['month'],
#     y=y_with_offset, # Use the y-values with random offset
#     mode='markers',
#     marker=dict(
#         symbol='diamond',  # Change marker to squares
#         size=monthly_stats_twitter['count'] / 3, # Using 'count' for size
#         sizemode='area',
#         sizeref=2. * monthly_stats_twitter['count'].max() / (40. ** 2),
#         sizemin=4,
#         color=monthly_stats_twitter['mean_pos_score'], # Color by mean_pos_score
#         colorscale='RdYlGn',
#         cmin=0.4, # Keep original cmin
#         cmax=0.8,  # Keep original cmax
#         showscale=True,
#         colorbar=dict(
#             title='Mean Pos-score',
#             x=0.5,
#             y=1.15,
#             xanchor='center',
#             yanchor='top',
#             orientation='h',
#             len=0.5
#         ),

#     ),
#     # visible='legendonly',
#     name='Twitter Sentiment',
#     yaxis='y3', # Left y-axis
#     customdata=monthly_stats_twitter[['mean_pos_score', 'std_correct_prob']].values, # Include std_correct_prob in customdata
#     hovertemplate=(
#         "<b>Month:</b> %{x|%Y-%m}<br>" +
#         "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
#         "<b>Std Dev (Correct Prob):</b> %{customdata[2]:.2f}<br>" +  # Show std dev in hover
#         "<b>Mean Pos-score:</b> %{customdata[0]:.2f}<br>" +
#         "<b>News Count:</b> %{marker.size}<extra></extra>"
#     )
# ))

# # --- Update Layout ---
# fig.update_layout(
#     xaxis=dict(title='Date'),
#     yaxis=dict(
#         title='S&P 500 ($)',
#         side='right', # S&P 500 on the right
#         #position = 0.95,
#         showgrid=False
#     ),
#     yaxis2=dict(
#         title='Installed Capacity Solar + Wind (MW)',
#         side='right', # Secondary right axis
#         overlaying='y', # Overlays the primary y-axis
#         anchor='free',  # Allows it to be positioned independently
#         autoshift=True, # THIS IS THE KEY! Automatically shifts to avoid overlap
#         showgrid=False,
#         automargin=True # Let Plotly adjust margin for this axis if needed

#     ),
#     yaxis3=dict(
#         title='Mean Probability of Correct Sentiment (%)',
#         side='left', # News Sentiment on the left
#         showgrid=True,
#         anchor='free', # Allow free positioning
#         overlaying='y', # Overlay on the primary y-axis
#         position=0 # Position of the left y-axis (0 is far left)
#     ),
#     legend=dict(x=0.01, y=0.99), # Legend position
#     height=600,
#     margin=dict(r=200),
#     # Make sure to set proper ranges if necessary or Plotly will auto-scale
#     # yaxis_range=[min_val_sp500, max_val_sp500],
#     # yaxis2_range=[min_val_renewables, max_val_renewables],
#     # yaxis3_range=[min_val_sentiment, max_val_sentiment],
# )

# st.plotly_chart(fig, use_container_width=True)



# #result_text = create_text_from_sent_analy_df(monthly_stats_twitter, monthly_stats_news ,filtered_sp500, filtered_df_energy)

# #st.write(result_text)

# # text = st.text_input(label='1', value=result_text)


# # if st.button("Play"):
# #if isinstance(result_text, str) and result_text.strip():
# #        tts = gTTS(result_text.strip(), lang="en")
# #        tts.save("output.mp3")
# #        st.audio("output.mp3", format="audio/mp3")
# #else:
# #        st.warning("Text field is empty or invalid.")

# st.markdown("---")


# def main():
#     """Main function to run the page"""
#     dashboard_info()
#     interactive_dashboard()

# # Run the page
# if __name__ == "__main__":
#     main()
# else:
#     # This runs when imported
#     interactive_dashboard()