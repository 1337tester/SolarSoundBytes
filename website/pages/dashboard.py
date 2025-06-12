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

####----OUR INTERACTIVE DASHBOARD----####
def interactive_dashboard():
    """Content for Dashboard page"""
    st.header("Our Interactive Dashboard")

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

    # Aggregiere nach Monat
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
        y=monthly_stats_news['mean_correct_prob'],
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
        y=monthly_stats_news['mean_correct_prob'], # Y-values are the same as the bubbles
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

    monthly_stats_twitter = monthly_stats_news

    # Generate random offsets for y-values
    random_offset = np.random.uniform(low=-0.2, high=0.2, size=len(monthly_stats_twitter))
    y_with_offset = monthly_stats_twitter['mean_correct_prob'] + random_offset * monthly_stats_twitter['mean_correct_prob']

    # --- News Sentiment Bubble Chart Trace (main trace) ---
    fig.add_trace(go.Scatter(
        x=monthly_stats_twitter['month'],
        y=y_with_offset, # Use the y-values with random offset
        mode='markers',
        marker=dict(
            symbol='diamond',  # Change marker to squares
            size=monthly_stats_twitter['count'] / 3, # Using 'count' for size
            sizemode='area',
            sizeref=2. * monthly_stats_twitter['count'].max() / (40. ** 2),
            sizemin=4,
            color=monthly_stats_twitter['mean_pos_score'], # Color by mean_pos_score
            colorscale='RdYlGn',
            cmin=0.4, # Keep original cmin
            cmax=0.8,  # Keep original cmax
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
        # visible='legendonly',
        name='Twitter Sentiment',
        yaxis='y3', # Left y-axis
        customdata=monthly_stats_twitter[['mean_pos_score', 'std_correct_prob']].values, # Include std_correct_prob in customdata
        hovertemplate=(
            "<b>Month:</b> %{x|%Y-%m}<br>" +
            "<b>Mean Correct Prob:</b> %{y:.2f}<br>" +
            "<b>Std Dev (Correct Prob):</b> %{customdata[2]:.2f}<br>" +  # Show std dev in hover
            "<b>Mean Pos-score:</b> %{customdata[0]:.2f}<br>" +
            "<b>News Count:</b> %{marker.size}<extra></extra>"
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



    #result_text = create_text_from_sent_analy_df(monthly_stats_twitter, monthly_stats_news ,filtered_sp500, filtered_df_energy)

    #st.write(result_text)

    # text = st.text_input(label='1', value=result_text)


    # if st.button("Play"):
    #if isinstance(result_text, str) and result_text.strip():
    #        tts = gTTS(result_text.strip(), lang="en")
    #        tts.save("output.mp3")
    #        st.audio("output.mp3", format="audio/mp3")
    #else:
    #        st.warning("Text field is empty or invalid.")

    st.markdown("---")


def main():
    """Main function to run the page"""
    dashboard_info()
    interactive_dashboard()

# Run the page
if __name__ == "__main__":
    main()
else:
    # This runs when imported
    interactive_dashboard()
