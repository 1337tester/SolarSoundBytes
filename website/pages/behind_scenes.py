import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def page_config():
    """Configure the page settings"""
    st.set_page_config(page_title="Behind the Scenes - SolarSoundBytes", layout="wide")

def header_section():
    """Display the main header and hero section"""
    st.title("🔧 Behind the Scenes")
    st.markdown("### How we built SolarSoundBytes: from raw data to AI-generated podcasts")
    st.markdown("""
        Welcome to our technical kitchen!</p>
        """, unsafe_allow_html=True)
    st.markdown("""
        Let us walk you through our journey of transforming years of
        public opinion, official news, market metrics, and historical events
        into a dynamic dashboard and time-traveling audio insights.
        </p>
        """, unsafe_allow_html=True)
    st.markdown("---")


####----Data Research Tab----####
def data_research_tab():
    """Content for the Data Research tab"""
    st.header("📊 Data Research & Collection")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("""Main Questions:""")
        st.write("""
                 - Where do we start?
                 - How do we find **trustworthy, high-quality data**?
                 - How can we ensure our data is **representative and unbiased**?
                 - Which **models are best suited** for sentiment analysis of tweets and news articles?
                 - How do we **visualize and interpret** the insights we uncover?
        """)

        st.subheader("Data Sources We Explored:")
        st.write("""
                To compare the sentiment of **news articles** to a broader **public sentiment**, we looked for a fitting twitter dataset.
                Although the **Climate Change Twitter Dataset (15 million tweets spanning over 13 years)** looked promising at first, we could not use it due to the lack of full-text tweets within.
                Since the vast majority of the most recent tweet_ids listed inside the Climate Change Twitter Dataset in GBR are no longer accessible, we abandoned our attempt to rehydrate this dataset.
                After extensive and unsuccessful further research for an alternative twitter dataset, we decided to create our own twitter dataset as input for a social media sentiment analysis using a scraping actor on console.apify.
                 """)
        st.write("### Twitter/X API")
        st.write("""
                To work with a user-friendly scraping GUI while keeping scraping costs below **40 USD/month**, the following scraper was chosen:
                - Tweet Scraper|$0.25/1K Tweets | Pay-Per Result | No Rate Limits.
                    Search Terms:
                    - Renewable Energy
                    - Energy Storage
                 """)

        st.write("### News Articles API")
        st.write("""
                 Online research for datasets of news-articles in the field of renewable energy technologies led us to the Cleantech Media Dataset by Anacode.
                 This dataset settled the timeframe of our data collection to **2022-01-02 to 2024-12-24**. Hoewever, after many tests, we found out that our data set was bias towards possitive sentiment.
                 Therefore, we decided to use the **News API** to collect news articles from **2022-01-02 to 2024-12-24**.
                 - GNews 49,00€/month: API results in JSON format via HTTP GET requests.
                    Search Terms:
                    - Renewable Energy
                    - Energy Storage
                 """)

        st.write("### Global Events")
        st.markdown("""
                Key global events with likely sentiment shifts were identified by conducting in-depth research
                using iterative ChatGPT-4.1 prompts. A summary is shown in the table in this [**link**](https://drive.google.com/file/d/16EfNelp-TF2qFCpzoPM9wJqfMXhSedhx/view?usp=sharing).
                """)


    with col2:
        st.markdown("**Data Metrics**")
        st.metric("Tweets Analyzed", "100,000+")
        st.metric("News Articles", "4,000+")
        st.metric("Total Words Processed", "4.5M+")  # New metric
        st.metric("Data Quality Score", "74.2%")

        # Data volume chart
        fig = go.Figure(data=go.Bar(
            x=['Tweets', 'News'],
            y=[100000, 4000],
            marker_color=['#1DA1F2', '#FF4500']
        ))
        fig.update_layout(
            title="Data Sources Volume",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Text volume chart (in millions of words)
        fig = go.Figure(data=go.Bar(
            x=['Tweets', 'News'],
            y=[10, 500],
            marker_color=['#1DA1F2', '#FF4500']
        ))
        fig.update_layout(
            title="Text Volume (Millions of Words)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


####----NLP Models Tab----#####
def nlp_models_tab():
    """Content for the NLP Models tab"""
    st.header("🤖 NLP Models")

    st.write("A model in **Machine Learning** is like a ***recipe in cooking***. It takes raw ingredients (data) and transforms them into a delicious dish (insights).")
    st.write("In our case, the ingredients are ***tweets and news articles***, and the dish is a ***podcast-style audio script*** that summarizes market sentiment.")
    st.write("We didn't just pick the first model we found.")

    st.subheader("Models We Tested:")
    col1, col2 = st.columns(2)
    with col1:
        models_tested = [
            "distilBERT",
            "twitter-RoBERTa",
            "nlptown",
            "VADER (NLTK)",
            "Gemma 3 / Vertex AI",
            "Custom DistilBERT Models (x3)"
        ]
        for i, model in enumerate(models_tested, 1):
            st.write(f"{i}. {model}")
    st.write("")

    st.subheader("Training, Test & Evaluate Models*")
    model_data = {
    "Model / Tool": [
        "distilBERT",
        "twitter-RoBERTa",
        "nlptown",
        "VADER (NLTK)",
        "Gemma 3 / Vertex AI",
        "Custom DistilBERT Models (x3)"
    ],
    "Type": [
        "Binary (Pos/Neg)",
        "Positive / Negative / Neutral",
        "Multilingual / Reviews",
        "Positive / Negative / Neutral",
        "Generative / Chat Model",
        "Fine-tuned Transformer"
    ],
    "Notes": [
        "★ Limited to 2 classes.",
        "✓ Better suited for Twitter (short text)",
        "✗ Optimized for product/movies reviews.",
        "✗ Ignores word context and syntax.",
        "✗ API unstable & evolving. Fine-tuning unsuccessful.",
        "✔ Fine-tuned 3x. Minimal preprocessing. Weighted for negatives — Data bias discovered."
    ],
    "Accuracy": [
        "High",
        "High",
        "Low",
        "Low",
        "Low",
        "Low\n(due to data bias)"
    ]}

    model_df = pd.DataFrame(model_data)
    st.table(model_df.reset_index(drop=True))


 # Map accuracy to numeric for plotting
    with col2:
        accuracy_map = {"High": 2, "Low": 1}
        df_plot = pd.DataFrame(model_data).copy()
        df_plot["Accuracy (Num)"] = df_plot["Accuracy"].apply(lambda x: accuracy_map["High"] if "High" in x else accuracy_map["Low"])
        df_plot["Accuracy Label"] = df_plot["Accuracy"].apply(lambda x: "High" if "High" in x else "Low")

        # Plot
        plt.figure(figsize=(7, 4))
        bar = sns.barplot(
        data=df_plot,
        y="Model / Tool",
        x="Accuracy (Num)",
        hue="Accuracy Label",
        dodge=False,
        palette={"High": "#2ECC71", "Low": "#E74C3C"}
        )
        bar.set_xlabel("Accuracy Level")
        bar.set_ylabel("Model / Tool")
        bar.set_xticks([1, 2])
        bar.set_xticklabels(["Low", "High"])
        plt.title("Sentiment Model Accuracy")
        plt.legend(title="Accuracy", loc="lower right")
        plt.tight_layout()
        st.pyplot(plt)


    st.success("🏆 **Winner: distilBERT Model** - Best balance of accuracy for both short and long text")

####----Pipeline Tab----#####
def pipeline_tab():
    """Content for the Pipeline tab"""
    st.header("⚙️ The Complete Pipeline")

    st.markdown("**From Raw Data to Insights: Our 6-Step Process**")

    # Pipeline steps
    pipeline_steps = [
        ("🔍 Data Scraping", "Collect tweets and news articles"),
        ("🧹 Text Cleaning", "Remove noise, normalize text"),
        ("🤖 Sentiment Analysis", "Apply our trained NLP model"),
        ("📊 Data Aggregation", "Merge sentiment insights from tweets and news with market (S&P 500) and economic (GDP) data to reveal the bigger picture."),
        ("📄 Script Generation", "Generate podcast-style summary content"),
        ("🎧 Text-to-Speech", "Convert to audio format")
    ]

    for i, (step, description) in enumerate(pipeline_steps, 1):
        with st.expander(f"Step {i}: {step}"):
            st.write(description)

            # Code examples for key steps
            if i == 1:
                st.code("""
            def scrape_financial_tweets(symbol, limit=100):
                tweets = api.search_tweets(
                    q=f"${symbol} OR #{symbol}",
                    count=limit,
                    lang='en'
                )
                return [tweet.text for tweet in tweets]
                            """, language='python')

            elif i == 2:
                st.code("""
            def preprocess_text(text):
                text = text.lower()
                text = re.sub(r".*'name':\s*'([^']+)'.*", r'\1', text)
                text = re.sub(r'^name\s+(.+?)\s+url\s+https.*', r'\1', text)
                tokens = text.split()
                return ' '.join(tokens)

                # Apply preprocessing to text columns
                text_columns = ['title', 'description', 'source', 'content']
                for col in text_columns:
                    df[f'Clean_{col.capitalize()}'] = df[col].apply(preprocess_text)

                # Clean and format date
                df['Clean_Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

                # Create clean dataframe by dropping original columns
                columns_to_drop = ['url', 'image', 'publishedAt', 'title', 'description', 'content', 'source', 'Date']
                df_clean = df.drop(columns=columns_to_drop)

                df_clean.head()
                            """, language='python')

            elif i == 3:
                st.code("""
                sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                df_sample = df_clean.sample(n=100, random_state=42).copy()

                df_sample[['Sentiment', 'Score']] = df_sample['Clean Article Text'].apply(analyze_sentiment_chunked)
                df_sample.head()
                    }
                """, language='python')

def audio_generation_tab():
    """Content for the Audio Generation tab"""
    st.header("🎵 Audio Generation Magic")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Text-to-Speech Pipeline:**")
        st.write("1. **Script Generation**: AI creates podcast-style narrative")
        st.write("2. **Voice Selection**: Choose appropriate voice characteristics")
        st.write("3. **Audio Processing**: Add background music and effects")
        st.write("4. **Quality Control**: Ensure natural speech patterns")

        st.markdown("**Audio Features:**")
        st.write("• Natural-sounding AI voices")
        st.write("• Background music integration")
        st.write("• Podcast-style formatting")
        st.write("• Multiple language support (planned)")

    with col2:
        st.markdown("**Sample Audio Script Generation:**")
        st.code("""
# Example of generated script
script = f'''
Welcome to your daily market sentiment briefing.

Today's analysis shows {sentiment_summary} sentiment
around {top_stocks}.

Let's dive into the details...
'''
        """, language='python')

        st.info("🎧 **Fun Fact**: Each podcast is unique and generated in real-time based on current market sentiment!")

def final_assembly_tab():
    """Content for the Final Assembly tab"""
    st.header("🚀 Final Assembly & Integration")

    st.markdown("**Bringing It All Together with Streamlit**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Architecture Overview:**")
        st.write("• **Frontend**: Streamlit web application")
        st.write("• **Backend**: Python data processing pipeline")
        st.write("• **Database**: Real-time sentiment data storage")
        st.write("• **APIs**: External data sources integration")
        st.write("• **Deployment**: Cloud-based hosting")

    with col2:
        st.markdown("**Technical Stack:**")
        tech_stack = {
            "Frontend": "Streamlit",
            "ML/NLP": "Transformers, scikit-learn",
            "Data Processing": "Pandas, NumPy",
            "Visualization": "Plotly, Matplotlib",
            "Audio": "Text-to-Speech APIs",
            "Deployment": "Streamlit Cloud"
        }

        for category, tech in tech_stack.items():
            st.write(f"• **{category}**: {tech}")

def main_tabs():
    """Display the main content tabs"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Research",
        "🤖 NLP Models",
        "⚙️ Pipeline",
        "🎵 Audio Generation",
        "🚀 Final Assembly"
    ])

    with tab1:
        data_research_tab()

    with tab2:
        nlp_models_tab()

    with tab3:
        pipeline_tab()

    with tab4:
        audio_generation_tab()

    with tab5:
        final_assembly_tab()

def whats_next_section():
    """Display the What's Next section"""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🎯 What's Next?</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Want to see our future plans? Check out our <strong>Upcoming Updates</strong> page!</p>", unsafe_allow_html=True)

def navigation_buttons():
    """Display navigation buttons"""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("home.py")
    with col3:
        if st.button("📊 Try Dashboard", use_container_width=True):
            st.switch_page("pages/app.py")
    with col4:
        if st.button("🚀 Future Features", use_container_width=True):
            st.switch_page("pages/upcoming.py")

def footer_section():
    """Display the footer with logo and credits"""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image('images/LeWagon_logo.png', width=150)
            st.markdown("<p style='text-align: center;'>Created by Le Wagon Data Science Batch #2012</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-style: italic;'>Built with ❤️ by the SolarSoundBytes team</p>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.info("📷 Logo not found")
            st.markdown("<p style='text-align: center;'>Created by Le Wagon Data Science Batch #2012</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-style: italic;'>Built with ❤️ by the SolarSoundBytes team</p>", unsafe_allow_html=True)

def render_behind_scenes():
    """Render function for importing into other pages"""
    header_section()
    main_tabs()
    whats_next_section()
    navigation_buttons()
    footer_section()

def main():
    """Main function to run the page"""
    page_config()
    render_behind_scenes()

# Run the page
if __name__ == "__main__":
    main()
else:
    # This runs when imported
    render_behind_scenes()
