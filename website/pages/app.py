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

from solarsoundbytes.compare_sent_analy.test_sentimental_analysis_calc import create_output_interface
from solarsoundbytes.text_creation.create_text import create_text_from_sent_analy_df
from solarsoundbytes.data_sp500 import get_sp500_df

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
df_sp500 = get_sp500_df()

# Quarterly GDP values
gdp = {
    '2022': 5200,
    '2023': 5500,
    '2024': 5750,
}


# Prepare S&P 500
df_sp500['Date'] = pd.to_datetime(df_sp500['Date'], format='%m/%d/%Y', errors='coerce')
df_sp500['Price'] = df_sp500['Price'].str.replace(',', '')
df_sp500['Price'] = pd.to_numeric(df_sp500['Price'], errors='coerce')
df_sp500['month'] = df_sp500['Date'].dt.to_period('M')
monthly_sp500 = df_sp500.groupby('month')['Price'].mean().reset_index()
monthly_sp500['month'] = monthly_sp500['month'].dt.to_timestamp()

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

# Prepare Tweets
df_tweets = df_twitter.copy()
df_tweets['createdAt'] = pd.to_datetime(df_tweets['createdAt'])
df_tweets['quarter'] = df_tweets['createdAt'].dt.to_period('Q')
df_tweets['three_month_block'] = (df_tweets['createdAt'].dt.year * 100 +
                                 ((df_tweets['createdAt'].dt.month - 1) // 3) * 3 + 1)
counts_twitter = df_tweets.groupby(['quarter', 'sentiment']).size().reset_index(name='count')
counts_twitter['quarter_start'] = counts_twitter['quarter'].dt.start_time

st.write(df_twitter.head())
st.write(df_news.head())

# Prepare news
df_news = df_news.copy()
df_news['Date Published'] = pd.to_datetime(df_news['Date Published'])
df_news['quarter'] = df_news['Date Published'].dt.to_period('Q')
df_news['three_month_block'] = (df_news['Date Published'].dt.year * 100 +
                                 ((df_news['Date Published'].dt.month - 1) // 3) * 3 + 1)
counts_news = df_news.groupby(['quarter', 'predicted_sentiment']).size().reset_index(name='count')
counts_news['quarter_start'] = counts_news['quarter'].dt.start_time


# Generate list of quarters (e.g. "2022 Q1", ..., "2024 Q4")
def generate_quarters(start_year, end_year):
    return [f"{year} Q{q}" for year in range(start_year, end_year + 1) for q in range(1, 5)]

quarters_list = generate_quarters(2022, 2024) # Umbenannt, um Konflikt mit 'quarters' Variable zu vermeiden

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
    # end_date sollte das Ende des Quartals sein, nicht der 1. des Folgemonats
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
# filtered_gdp wird im aktuellen Plot nicht direkt verwendet, da df_gdp_yearly genutzt wird.
# filtered_gdp = df_gdp[(df_gdp['quarter_start'] >= start_date) & (df_gdp['quarter_start'] <= end_date)]

# --- Plot-Erstellung ---
fig = go.Figure()

# S&P 500 Linie
fig.add_trace(go.Scatter(
    x=filtered_sp500['month'], y=filtered_sp500['Price'],
    name='S&P 500',
    yaxis='y1',
    mode='lines+markers',
    line=dict(color='blue')
))

# GDP Linien pro Jahr, auf den ausgewählten Datumsbereich zugeschnitten
first_gdp = True

for _, row in df_gdp_yearly.iterrows():
    year = row['year']
    value = row['gdp']
    year_start = pd.to_datetime(f"{year}-01-01")
    year_end = pd.to_datetime(f"{year}-12-31")

    # Linie auf den ausgewählten Bereich zuschneiden
    line_start = max(start_date, year_start)
    line_end = min(end_date, year_end)

    # Nur zeichnen, wenn die Linie im sichtbaren Bereich liegt
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

# --- NEUE BUBBLE-LOGIK FÜR INEINANDERLIEGENDE KREISE ---

# Aggregieren der Zählungen pro Quartalsbeginn
# Gruppieren nach 'quarter_start' und Pivotieren nach 'sentiment'
agg_counts_bubbles_twitter = filtered_counts_twitter.pivot_table(
    index='quarter_start',
    columns='sentiment',
    values='count',
    fill_value=0
).reset_index()

agg_counts_bubbles_news = filtered_counts_news.pivot_table(
    index='quarter_start',
    columns='predicted_sentiment',
    values='count',
    fill_value=0
).reset_index()

st.write(agg_counts_bubbles_news)
# Berechne die erforderlichen Zählungen für die Kreise
agg_counts_bubbles_twitter['total_count'] = agg_counts_bubbles_twitter['positive'] + agg_counts_bubbles_twitter['neutral'] + agg_counts_bubbles_twitter['negative']
agg_counts_bubbles_twitter['negative_neutral_count'] = agg_counts_bubbles_twitter['neutral'] + agg_counts_bubbles_twitter['negative']
# 'negative' ist bereits als Spalte vorhanden
agg_counts_bubbles_news['total_count'] = agg_counts_bubbles_news['positive']    # + agg_counts_bubbles_news['neutral'] + agg_counts_bubbles_news['negative']
agg_counts_bubbles_news['negative_neutral_count'] = 0   #agg_counts_bubbles_news['neutral'] + agg_counts_bubbles_news['negative']

# Bestimmen der Y-Position der Blasen. Sie sollte konstant sein.
# Passen Sie diesen Wert an, damit die Blasen gut sichtbar sind und andere Plot-Elemente nicht überlappen.
# Versuchen Sie einen Wert, der unterhalb des S&P 500-Bereichs liegt, aber noch im Diagramm sichtbar ist.
# Eine Annahme für den unteren Rand der S&P 500 Achse könnte hilfreich sein.
bubble_y_position_twitter = filtered_sp500['Price'].min() - 2000 if not filtered_sp500.empty else 1000
bubble_y_position_news = filtered_sp500['Price'].min() - 1500 if not filtered_sp500.empty else 1000

# Maximale Größe für die Skalierung der Blasen
# Die sizeref sollte sich auf die größte mögliche Blase (total_count) beziehen.
# Stellen Sie sicher, dass max_total_count mindestens 1 ist, um Division durch Null zu vermeiden.
max_total_count_twitter = agg_counts_bubbles_twitter['total_count'].max() if not agg_counts_bubbles_twitter.empty else 1
# Ihre ursprüngliche size_max war 40, daher sizeref-Berechnung basierend darauf
sizeref_calc_twitter = 2. * max_total_count_twitter / (40. ** 2)
sizeref_twitter = sizeref_calc_twitter if sizeref_calc_twitter > 0 else 1 # Sicherstellen, dass sizeref nicht 0 ist


max_total_count_news = agg_counts_bubbles_news['total_count'].max() if not agg_counts_bubbles_news.empty else 1
# Ihre ursprüngliche size_max war 40, daher sizeref-Berechnung basierend darauf
sizeref_calc_news = 2. * max_total_count_news / (40. ** 2)
sizeref_news = sizeref_calc_news if sizeref_calc_news > 0 else 1 # Sicherstellen, dass sizeref nicht 0 ist


# Hinzufügen der drei konzentrischen Kreise als separate Spuren
# Wichtig: Die größte Blase zuerst hinzufügen, dann die kleineren, damit sie sich überlappen.

# Äußerster Kreis: Gesamtzahl der Tweets
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

    # Mittlerer Kreis: Negative und neutrale Tweets
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

    # Innerster Kreis: Negative Tweets
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

    # Mittlerer Kreis: neutrale News
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

    # Innerster Kreis: Negative News
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


# with col1:
#     st.plotly_chart(fig, use_container_width=True)


# with col2:
#     selected_sentiment = st.radio("Wähle ein Sentiment", ['positive', 'neutral', 'negative'])

#     # Dummy-Zeitreihe generieren
#     dummy_dates = pd.date_range(start=start_date, end=end_date, freq='M')
#     dummy_values = np.random.randn(len(dummy_dates)).cumsum()

#     fig_dummy = go.Figure()
#     fig_dummy.add_trace(go.Scatter(
#         x=dummy_dates, y=dummy_values,
#         mode='lines+markers',
#         line=dict(color='purple'),
#         name='Dummy Curve'
#     ))

#     fig_dummy.update_layout(title=f"Dummy-Kurve für: {selected_sentiment.capitalize()}")
#     st.plotly_chart(fig_dummy, use_container_width=True)





################################################################################
################# second plot #######################################################
################################################################################





# Twitter-Daten monatlich gruppieren
df_twitter['createdAt'] = pd.to_datetime(df_twitter['createdAt'])
df_twitter['sentiment'] = df_twitter['sentiment'].str.lower()
df_twitter['date'] = df_twitter['createdAt'].dt.to_period('M').dt.to_timestamp()

# Gruppieren nach Monat und Sentiment, dann aggregieren
agg_df_twitter = df_twitter.groupby(['date', 'sentiment']).agg(
    count=('sentiment', 'size'),
    confidence_score_mean=('confidence score', 'mean')
).reset_index()

# Eindeutige numerische Reihenfolge für Zeit → date_order
unique_dates_twitter = agg_df_twitter[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
unique_dates_twitter['date_order'] = unique_dates_twitter.index + 1  # 1-basiert

agg_df_twitter = agg_df_twitter.merge(unique_dates_twitter, on='date', how='left')

# Nur alle 6 Monate anzeigen
tick_df_twitter = unique_dates_twitter[unique_dates_twitter['date'].dt.month.isin([1, 7])]
tickvals_twitter = tick_df_twitter['date_order'].tolist()
ticktext_twitter = tick_df_twitter['date'].dt.strftime('%b %Y').tolist()

# News-Daten monatlich gruppieren
df_news['Date Published'] = pd.to_datetime(df_news['Date Published'])
df_news['predicted_sentiment'] = df_news['predicted_sentiment'].str.lower()
df_news['date'] = df_news['Date Published'].dt.to_period('M').dt.to_timestamp()

# Gruppieren nach Monat und Sentiment, dann aggregieren
agg_df_news = df_news.groupby(['date', 'predicted_sentiment']).agg(
    count=('predicted_sentiment', 'size'),
    confidence_score_mean=('confidence_score', 'mean')
).reset_index()

# Eindeutige numerische Reihenfolge für Zeit → date_order
unique_dates_news = agg_df_news[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
unique_dates_news['date_order'] = unique_dates_news.index + 1  # 1-basiert

agg_df_news = agg_df_news.merge(unique_dates_news, on='date', how='left')

# Quelle hinzufügen
agg_df_news['source'] = 'news'
agg_df_twitter['source'] = 'twitter'

# Spalte umbenennen für Konsistenz
agg_df_news.rename(columns={'predicted_sentiment': 'sentiment'}, inplace=True)

# Kombinierte Daten
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

# Achsenbeschriftungen
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



# st.markdown("## Sentimentverlauf – Bubble-Chart")

# # 1. Bubble-Chart normal anzeigen (wie früher)

# # 2. Klickereignisse unsichtbar abfangen
# selected_points = plotly_events(fig, click_event=True, override_height=600)

# # 3. Wenn geklickt → Detailanzeige
# if selected_points:
#     clicked_date = pd.to_datetime(selected_points[0]['customdata'][0]).date()
#     st.markdown(f"### Detailansicht für den {clicked_date.strftime('%d.%m.%Y')}")

#     # Filter auf diesen Tag
#     df_day = result[result['date'].dt.date == clicked_date]

#     if df_day.empty:
#         st.warning("Für diesen Tag sind keine Einträge vorhanden.")
#     else:
#         df_day = df_day.copy()
#         df_day['hour'] = df_day['date'].dt.hour

#         fig_hist = px.histogram(df_day, x='hour', color='sentiment',
#                                 title="Anzahl der Einträge nach Stunde", nbins=24,
#                                 labels={'hour': 'Stunde'})
#         fig_pie = px.pie(df_day, names='sentiment', title="Sentiment-Verteilung am Tag")
#         fig_scatter = px.scatter(df_day, x='date', y='sentiment', color='sentiment',
#                                  title="Sentiment-Timeline (nach Uhrzeit)")

#         st.plotly_chart(fig_hist, use_container_width=True)
#         st.plotly_chart(fig_pie, use_container_width=True)
#         st.plotly_chart(fig_scatter, use_container_width=True)




# df_sp500 = get_sp500_df()
# df_sp500['Date'] = pd.to_datetime(df_sp500['Date'], format='%m/%d/%Y', errors='coerce')
# df_sp500 = df_sp500.sort_values(by='Date').reset_index(drop=True)
# st.write(df_sp500.head())


# fig = px.line(
#     df_sp500,
#     x=df_sp500['Date'],
#     y=df_sp500['Price']
# )

# st.plotly_chart(fig)



# df_merged = df_long.merge(df_sp500[['Date', 'Price']], left_on='date', right_on='Date', how='left')

# # Plot
# fig = px.scatter(
#     df_merged,
#     x="date",
#     y="Price",
#     color="sentiment",
#     size="count",
#     animation_frame="date_order",  # optional
#     hover_data=["sentiment", "count", "Price", "date"],
#     title="Sentiment-Bubbles auf Preisverlauf (Y-Achse = S&P 500 Preis)",
#     size_max=40
# )

# # Layout-Optimierung
# fig.update_layout(
#     xaxis_title="Datum",
#     yaxis_title="S&P 500 Preis",
#     legend_title="Sentiment",
#     height=600
# )

# # In Streamlit anzeigen
# st.plotly_chart(fig, use_container_width=True)



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


# result_twitter, result_news = create_output_interface()
# df_twitter = pd.DataFrame(result_twitter)
# df_news = pd.DataFrame(result_news)

# # # Plotly-figure
# fig = go.Figure()

# # left y-axis
# fig.add_trace(go.Scatter(
#     x=df_twitter['month_year'], y=round(df_twitter['multiplication'],6), name='Tweets',
#     yaxis='y1', mode='lines+markers'
# ))

# # left y-axis
# fig.add_trace(go.Scatter(
#     x=df_news['month_year'], y=round(df_news['multiplication'],6), name='News',
#     yaxis='y1', mode='lines+markers'
# ))

# # #axis layout
# fig.update_layout(
#     #title='Zwei Y-Achsen Diagramm',
#     xaxis=dict(title='Month'),
#     yaxis=dict(title='Sentiment Analysis', side='left'),
#     # yaxis2=dict(title='inst. el. capacity renewable energies []', overlaying='y', side='right'),
#     legend=dict(orientation='h',
#         yanchor='bottom',
#         y=-0.3,  # Abstand zur Grafik; ggf. anpassen
#         xanchor='center',
#         x=0.5
#     ),
#     margin=dict(b=100)  # extra Platz unten
# )

# st.plotly_chart(fig, use_container_width=True)


# # # Plotly-figure
# fig = go.Figure()

# # left y-axis
# fig.add_trace(go.Scatter(
#     x=df_twitter['month_year'], y=round(df_twitter['mean_pos'],6)*100, name='Tweets_pos',
#     yaxis='y1', mode='lines+markers'
# ))

# # left y-axis
# fig.add_trace(go.Scatter(
#     x=df_news['month_year'], y=round(df_news['mean_pos'],6)*100, name='News_pos',
#     yaxis='y1', mode='lines+markers'
# ))

# # left y-axis
# fig.add_trace(go.Scatter(
#     x=df_twitter['month_year'], y=round(df_twitter['mean_neg'],6)*100, name='Tweets_neg',
#     yaxis='y2', mode='lines+markers'
# ))

# # left y-axis
# fig.add_trace(go.Scatter(
#     x=df_news['month_year'], y=round(df_news['mean_neg'],6)*100, name='News_neg',
#     yaxis='y2', mode='lines+markers'
# ))

# # #axis layout
# fig.update_layout(
#     xaxis=dict(title='Month'),
#     yaxis=dict(title='pos Sentiment Analysis [%]', side='left'),
#     yaxis2=dict(title='neg Sentiment Analysis [%]', overlaying='y', side='right'),
#     legend=dict(orientation='h',
#         yanchor='bottom',
#         y=-0.3,  # Abstand zur Grafik; ggf. anpassen
#         xanchor='center',
#         x=0.5
#     ),
#     margin=dict(b=100)  # extra Platz unten
# )

# st.plotly_chart(fig, use_container_width=True)

# # with st.form(key='params_for_api'):

# #     pickup_date = st.date_input('pickup datetime', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
# #     pickup_time = st.time_input('pickup datetime', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
# #     pickup_datetime = f'{pickup_date} {pickup_time}'
# #     pickup_longitude = st.number_input('pickup longitude', value=40.7614327)
# #     pickup_latitude = st.number_input('pickup latitude', value=-73.9798156)
# #     dropoff_longitude = st.number_input('dropoff longitude', value=40.6413111)
# #     dropoff_latitude = st.number_input('dropoff latitude', value=-73.7803331)
# #     passenger_count = st.number_input('passenger_count', min_value=1, max_value=8, step=1, value=1)

# #     st.form_submit_button('Make prediction')

# # params = dict(
# #     pickup_datetime=pickup_datetime,
# #     pickup_longitude=pickup_longitude,
# #     pickup_latitude=pickup_latitude,
# #     dropoff_longitude=dropoff_longitude,
# #     dropoff_latitude=dropoff_latitude,
# #     passenger_count=passenger_count)

# # wagon_cab_api_url = 'https://taxifare.lewagon.ai/predict'
# # response = requests.get(wagon_cab_api_url, params=params)

# # prediction = response.json()

# # pred = prediction['fare']

# # st.header(f'Fare amount: ${round(pred, 2)}')
