# -------------------------  previous code
import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
    "Global installed solar PV capacity surpasses 1 terawatt milestone": "2023-11-30",
    "COP28 concludes with historic agreement to transition away from fossil fuels": "2023-12-13"
}

# Convert event dates to datetime objects
EVENT_DATES = {event: pd.to_datetime(date) for event, date in GLOBAL_EVENTS.items()}

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

# --- Streamlit UI ---
st.set_page_config(page_title="Monthly Sentiment Visualization", layout="wide")
st.title("Monthly Sentiment Consensus: Articles vs Tweets")

st.markdown("""
- **Circles**: Articles  
- **Rhombi**: Tweets  
- **Y-axis**: Sentiment consensus (higher = more agreement, lower = more disagreement)  
- **Color**: Red (negative) to Green (positive)  
- **Size**: Number of texts (articles/tweets)  
""")

# Controls
df = generate_mock_data()
months = df['month'].dt.strftime('%Y-%m').unique()

# Event Selection
st.sidebar.header("Event Selection")
selected_event = st.sidebar.selectbox(
    "Select Global Event:",
    options=["None"] + [f"{date.strftime('%Y-%m-%d')} {event}" for event, date in EVENT_DATES.items()]
)

# --- Time Window Selection ---
if selected_event and selected_event != "None":
    # Extract the event name from the selected_event string
    # The format is "YYYY-MM-DD Event Name"
    event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
    event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])
    # Set window to ±1 day around the event date
    start_date = event_date - pd.Timedelta(days=1)
    end_date = event_date + pd.Timedelta(days=1)
    # Find the closest indices in the months array for start_date and end_date
    months_dt = pd.to_datetime(months, format='%Y-%m')
    # Find the first month >= start_date and last month <= end_date
    start_idx = np.searchsorted(months_dt, start_date, side='left')
    end_idx = np.searchsorted(months_dt, end_date, side='right') - 1
    # Clip to valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(months) - 1, end_idx)
    # Ensure valid window
    if start_idx > end_idx or start_idx >= len(months) or end_idx < 0:
        df_window = pd.DataFrame(columns=df.columns)  # empty DataFrame
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

# Prepare months_with_data for animation (moved this definition here, after df_window is defined)
months_with_data = []
if not df_window.empty:
    months_with_data = sorted(set(
        df_window[df_window['source'] == 'article']['month'].unique().tolist() +
        df_window[df_window['source'] == 'tweet']['month'].unique().tolist()
    ))

# Animation Controls
st.sidebar.header("Animation Controls")
show_animation = st.sidebar.checkbox("Show animation (month by month)", value=False)

# Metric Selection
st.sidebar.header("Metric Overlays")
selected_metrics = st.sidebar.multiselect(
    "Select metrics to overlay:",
    options=['Solar Investment', 'Oil Investment', 'Renewable Energy Jobs', 'Carbon Emissions'],
    default=[]
)

# Generate metric data
metric_df = generate_metric_data(pd.to_datetime(months))

# --- Plotly Figure ---
fig = go.Figure()

# 1. Initialize two empty traces as placeholders. These will be updated by frames.
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
            opacity=1.0 # Will be dynamically set by frames
        ),
        name=source.capitalize(),
        legendgroup=source,
        showlegend=True,
        text=[],
        hoverinfo='text',
    ))

# Define colors for metric overlays early so they are available for initialization
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown'] # Extend colors for more metrics

# Initialize metric traces if animating. If not animating, they are added later.
if show_animation:
    for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
        trace_yaxis_value = f'y{metric_idx + 2}' # y2, y3, etc.
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name=metric,
            line=dict(color=color),
            yaxis=trace_yaxis_value,
            showlegend=True, # Ensure legend is shown
            legendgroup=metric # Group legend entries
        ))

# 2. Populate the initial traces based on animation state (static or first frame of animation)
if not show_animation: # If not animating, show all data in the current window
    for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
        d_all_data = df_window[df_window['source'] == source]
        fig.data[source_idx].x = d_all_data['month'].tolist() if not d_all_data.empty else []
        fig.data[source_idx].y = d_all_data['std_sentiment'].tolist() if not d_all_data.empty else []
        fig.data[source_idx].marker.size = np.sqrt(d_all_data['count'])*3 if not d_all_data.empty else []
        fig.data[source_idx].marker.color = [sentiment_color(v) for v in d_all_data['mean_sentiment']] if not d_all_data.empty else []
        fig.data[source_idx].text = [
            f"{source.capitalize()}<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}" 
            for m, s, c, cnt in zip(d_all_data['month'], d_all_data['mean_sentiment'], d_all_data['std_sentiment'], d_all_data['count'])
        ] if not d_all_data.empty else []
elif months_with_data: # If animating and there's data to animate, initialize with first month's data
    first_month = months_with_data[0]
    for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
        # Initialize with cumulative data up to the first month for the trace effect
        d_first_month = df_window[(df_window['source'] == source) & (df_window['month'] <= first_month)]
        fig.data[source_idx].x = d_first_month['month'].tolist() if not d_first_month.empty else []
        fig.data[source_idx].y = d_first_month['std_sentiment'].tolist() if not d_first_month.empty else []
        fig.data[source_idx].marker.size = np.sqrt(d_first_month['count'])*3 if not d_first_month.empty else []
        # Prepare marker colors and opacity based on the trace logic for the initial frame
        marker_colors = []
        marker_opacities = []
        if not d_first_month.empty:
            for i, row in d_first_month.iterrows():
                marker_colors.append(sentiment_color(row['mean_sentiment']))
                if row['month'] < first_month: # Faded for past points in trace mode
                    marker_opacities.append(0.3)
                else: # Opaque for current point
                    marker_opacities.append(1.0)
        fig.data[source_idx].marker.color = marker_colors
        fig.data[source_idx].marker.opacity = marker_opacities
        fig.data[source_idx].text = [
            f"{source.capitalize()}<br>Month: {m.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}" 
            for m, s, c, cnt in zip(d_first_month['month'], d_first_month['mean_sentiment'], d_first_month['std_sentiment'], d_first_month['count'])
        ] if not d_first_month.empty else []
    
    # Also initialize metric traces with data up to the first month
    for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
        # Ensure the trace exists. The sentiment traces are at index 0 and 1.
        # Metric traces will start from index 2.
        metric_trace_idx = 2 + metric_idx
        if metric_trace_idx < len(fig.data):
            d_metric_first_month = metric_df[metric_df['month'] <= first_month]
            fig.data[metric_trace_idx].x = d_metric_first_month['month'].tolist() if not d_metric_first_month.empty else []
            fig.data[metric_trace_idx].y = d_metric_first_month[metric].tolist() if not d_metric_first_month.empty else []

# Add metric overlays

# Prepare dynamic y-axis definitions and trace assignments
metric_yaxis_layout_config = {} # For layout definition (yaxis2, yaxis3)
current_plot_yaxis_id = 2 # Start from 2 for 'y2'

for metric, color in zip(selected_metrics, colors):
    layout_axis_key = f'yaxis{current_plot_yaxis_id}' # e.g., 'yaxis2' for layout
    trace_yaxis_value = f'y{current_plot_yaxis_id}' # e.g., 'y2' for trace

    metric_yaxis_layout_config[layout_axis_key] = dict(
        title=metric,
        overlaying='y', # This means it overlays the primary 'y' axis
        side='right',
        showgrid=False,
        automargin=True, # Automatically adjust margin to prevent overlap
        anchor='free',   # Allow free positioning
        # Calculate position to avoid overlap, 1 is far right, 0.05 is an offset
        position=1 - (0.07 * (current_plot_yaxis_id - 1)) # Increased offset for better spacing
    )
    # Only add metric traces if not animating, as they are initialized above if animating
    if not show_animation:
        fig.add_trace(go.Scatter(
            x=metric_df['month'],
            y=metric_df[metric],
            mode='lines',
            name=metric,
            line=dict(color=color),
            yaxis=trace_yaxis_value,
            showlegend=True, # Ensure legend is shown
            legendgroup=metric # Group legend entries
        ))
    current_plot_yaxis_id += 1

# Add event line if event is selected
if selected_event and selected_event != "None" and not df_window.empty:
    event_name_only = selected_event.split(' ', 1)[1] if ' ' in selected_event else selected_event
    event_date = pd.to_datetime(GLOBAL_EVENTS[event_name_only])
    x_type = type(df_window['month'].iloc[0])
    event_month = pd.Timestamp(event_date.year, event_date.month, 1)
    event_month = x_type(event_month)  # Cast to same type as x-axis
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

# Animation
if show_animation:
    frames = []
    for m_idx, m in enumerate(months_with_data):
        frame_updates_for_traces = [] # This will hold update dicts for fig.data[0], fig.data[1] etc.
        for source_idx, (source, shape) in enumerate(zip(['article', 'tweet'], ['circle', 'diamond'])):
            # if show_trace:
                # Cumulative data for 'show trace' effect (always on now)
            d_for_frame = df_window[(df_window['source'] == source) & (df_window['month'] <= m)]
            # else:
                # Only current month data for single symbol effect
                # d_for_frame = df_window[(df_window['source'] == source) & (df_window['month'] == m)]

            # Prepare marker colors and opacity based on show_trace and current month
            marker_colors = []
            marker_opacities = []
            if not d_for_frame.empty:
                for i, row in d_for_frame.iterrows():
                    marker_colors.append(sentiment_color(row['mean_sentiment']))
                    # if show_trace and row['month'] < m: # Faded for past points in trace mode (always on now)
                    if row['month'] < m: 
                        marker_opacities.append(0.3)
                    else: # Opaque for current point or in single-symbol mode
                        marker_opacities.append(1.0)
            
            # Create a dictionary of updates for the current trace (by index)
            frame_updates_for_traces.append({
                'x': d_for_frame['month'].tolist() if not d_for_frame.empty else [],
                'y': d_for_frame['std_sentiment'].tolist() if not d_for_frame.empty else [],
                'marker': {
                    'size': np.sqrt(d_for_frame['count'])*3 if not d_for_frame.empty else [],
                    'color': marker_colors, # Use dynamically set colors
                    'symbol': shape, 
                    'line': dict(width=1, color='black'), 
                    'opacity': marker_opacities # Use dynamically set opacities
                },
                'text': [
                    f"{source.capitalize()}<br>Month: {mo.strftime('%Y-%m')}<br>Mean Sentiment: {s:.2f}<br>Consensus: {c:.2f}<br>Count: {cnt}" 
                    for mo, s, c, cnt in zip(d_for_frame['month'], d_for_frame['mean_sentiment'], d_for_frame['std_sentiment'], d_for_frame['count'])
                ] if not d_for_frame.empty else [],
            })
        
        # Add updates for metric traces
        for metric_idx, (metric, color) in enumerate(zip(selected_metrics, colors)):
            metric_trace_idx = 2 + metric_idx # Sentiment traces are at index 0 and 1
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
    else:
        st.warning("No data available for animation in the selected window.")

# Update layout
# The yaxis2, yaxis3 etc. will be added dynamically by metric_yaxis_layout_config

# Calculate dynamic x-axis tick values
visible_months = df_window['month'].unique()
if len(visible_months) > 12:
    # Select up to 12 evenly spaced months
    # np.linspace returns evenly spaced numbers over a specified interval.
    # We convert these to integers to use as indices for visible_months.
    indices_to_show = np.linspace(0, len(visible_months) - 1, 12, dtype=int)
    display_tickvals = [visible_months[i] for i in indices_to_show]
else:
    display_tickvals = visible_months

fig.update_layout(
    xaxis=dict(title='Month', tickformat='%Y-%m', tickvals=display_tickvals),
    yaxis=dict( # This is the primary 'y' or 'y1' axis for sentiment
        title='Sentiment Consensus (Std Dev)',
        autorange='reversed',
        side='left', # Explicitly set sentiment axis to the left
    ),
    legend=dict(title='Source'),
    height=900,
    margin=dict(l=40, r=150, t=60, b=300), # Fixed and increased bottom margin to ensure space for annotations
    plot_bgcolor='white',
    **metric_yaxis_layout_config # Unpack the dynamically created y-axis configurations for the layout
)

# Add arrows for global events if they are within the displayed x-axis range
# We'll use the original GLOBAL_EVENTS for event names and EVENT_DATES for their datetime objects

# Get the currently displayed x-axis range from df_window
if not df_window.empty:
    current_x_min = df_window['month'].min()
    current_x_max = df_window['month'].max()
else:
    current_x_min = None
    current_x_max = None

# Iterate through all global events
for event_name_full, event_date in EVENT_DATES.items():
    # Check if the event date is within the current visible range
    if current_x_min and current_x_max and current_x_min <= event_date <= current_x_max:
        fig.add_annotation(
            x=event_date, # X-coordinate of the arrow tip (event date)
            y=-0.05,  # Y-coordinate of the arrow tip (on x-axis in paper units)
            xref="x", # X-reference to the x-axis
            yref="paper", # Y-reference to the entire plot area (0 to 1)
            text=SHORT_EVENT_LABELS.get(event_name_full, event_name_full), # Use short label
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="darkgrey",
            ax=0, # X-coordinate of the arrow tail (aligned with tip horizontally)
            ay=100, # Y-coordinate of the arrow tail (negative value makes arrow point upwards from text)
            textangle=-90, # Vertical text
            valign="bottom", # Align text to the bottom of its bounding box (flows upwards on left when rotated -90)
            font=dict(color="darkgrey", size=10)
        )

st.plotly_chart(fig, use_container_width=True)
