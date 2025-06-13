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
                To compare the sentiment of **news articles** to a broader **public sentiment**, we looked for a fitting twitter and news article datasets.
                Both the **Climate Change Twitter Dataset (15 million tweets spanning over 13 years)** and the **Cleantech Media Dataset by Anacode** looked promising at first, but we could not use them due to several limitations:
                - The lack of full-text tweets in the dataset.
                - News articles were bias towards positive sentiment.""")
        st.write("""
                As we were advancing int our process, the Cleantech Media Dataset settled the timeframe of our data collection to **2022-01-02 to 2024-12-24**.
                After extensive and unsuccessful further research for alternative datasets, we decided to create our own datasets for both, tweets and news articles
                for a social media sentiment analysis using a scraping actor on [console.apify](https://console.apify.com/).
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
                 The **News API** was our main tool to collect news articles covering many
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
 #   with col2:
 #       accuracy_map = {"High": 2, "Low": 1}
 #       df_plot = pd.DataFrame(model_data).copy()
 #       df_plot["Accuracy (Num)"] = df_plot["Accuracy"].apply(lambda x: accuracy_map["High"] if "High" in x else accuracy_map["Low"])
 #       df_plot["Accuracy Label"] = df_plot["Accuracy"].apply(lambda x: "High" if "High" in x else "Low")
#
 #       # Plot
 #       plt.figure(figsize=(7, 4))
 #       bar = sns.barplot(
 #       data=df_plot,
 #       y="Model / Tool",
 #       x="Accuracy (Num)",
 #       hue="Accuracy Label",
 #       dodge=False,
 #       palette={"High": "#2ECC71", "Low": "#E74C3C"}
 #       )
 #       bar.set_xlabel("Accuracy Level")
 #       bar.set_ylabel("Model / Tool")
 #       bar.set_xticks([1, 2])
 #       bar.set_xticklabels(["Low", "High"])
 #       plt.title("Sentiment Model Accuracy")
 #        plt.legend(title="Accuracy", loc="lower right")
 #       plt.tight_layout()
 #      st.pyplot(plt)


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
        ("📄 Text Generation", "Generate podcast-style summary content"),
        ("🎧 Text-to-Speech", "Convert to audio format")
    ]

    for i, (step, description) in enumerate(pipeline_steps, 1):
        with st.expander(f"Step {i}: {step}"):
            st.write(description)

            # Code examples for key steps
            if i == 1:
                st.markdown("[**Twitter API user story**](https://drive.google.com/file/d/1uVTl7SvQNJE00I0GaDez2XCjw0byp4j7/view?usp=sharing)")
                st.code("""
            def extract_tweet_data(tweet, reference_date):
    def safe_get(dct, key, default=None):
        return dct.get(key, default)

    def get_user_mentions(entities):
        mentions = safe_get(entities, "user_mentions", [])
        id_strs = "~~".join([m.get("id_str", "") for m in mentions])
        indices_0 = mentions[0]["indices"][0] if len(mentions) > 0 else None
        indices_1 = mentions[1]["indices"][1] if len(mentions) > 1 else None
        name = "~~".join([m.get("name", "") for m in mentions])
        screen_name = "~~".join([m.get("screen_name", "") for m in mentions])
        return id_strs, indices_0, indices_1, name, screen_name

    entities = tweet.get("entities", {})
    user_mentions = get_user_mentions(entities)
                            """, language='python')

                st.markdown("[**News Articles API user story**](https://drive.google.com/file/d/1LsZA_0e8LvhuxZZMg6myaI6gNTh1d0-e/view?usp=sharing)")
                st.code("""
                        def articles_api_2_csv(t_start_str: str, t_end_str: str, query: str, query_subdivisions: int = 1):

    # --------------------- API call ---------------------

    # load API key and plan-specific max_n_articles from .env file
    load_dotenv()
    API_KEY = os.getenv("GNEWS_API_KEY")
    MAX_N_ARTICLES = os.getenv("GNEWS_MAX_N_ARTICLES")

    if not API_KEY:
        raise ValueError("GNEWS_API_KEY not found in environment variables.")
                        """)

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
                sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                df_sample = df_clean.sample(n=100, random_state=42).copy()

                df_sample[['Sentiment', 'Score']] = df_sample['Clean Article Text'].apply(analyze_sentiment_chunked)
                df_sample.head()
                    }
                """, language='python')

            elif i == 5:
                st.code("""
                client = OpenAI(api_key = api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",  #
                    messages=[
                        {"role": "system", "content": "You are a data-analytical journalist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )""", language='python')

            elif i == 6:
                st.code("""
                if st.button("Play"):
                if isinstance(text, str) and text.strip():
                    tts = gTTS(text.strip(), lang="en")
                    tts.save("output.mp3")
                    st.audio("output.mp3", format="audio/mp3")
                else:
                    st.warning("Text field is empty or invalid.")
                """, language='python')


    """Content for the Final Assembly tab"""
    st.header("🚀 Final Assembly & Integration")

    st.subheader("Bringing It All Together with Streamlit")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Architecture Overview:**")
        st.write("• **Frontend**: Streamlit web application")
        st.write("• **Backend**: Python data processing pipeline")
        st.write("• **Database**: Sentiment anailysis")
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
    tab1, tab2, tab3, = st.tabs([
        "📊 Data Research",
        "🤖 NLP Models",
        "⚙️ Pipeline"
    ])

    with tab1:
        data_research_tab()

    with tab2:
        nlp_models_tab()

    with tab3:
        pipeline_tab()

def whats_next_section():
    """Display the What's Next section"""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🎯 What's Next?</h2>", unsafe_allow_html=True)
def navigation_buttons():
    """Display navigation buttons"""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        st.link_button("🏠 Back to Home", "home.py", use_container_width=True)
    with col3:
        st.link_button("📊 Try Dashboard", "pages/app.py", use_container_width=True)
    with col4:
        st.link_button("🚀 Future Features", "https://github.com/FadriPestalozzi/SolarSoundBytes")

def footer_section():
    """Display the footer with logo and credits"""
    st.markdown("---")
    # Use one wide column for centering
    col1, = st.columns([1])
    with col1:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="images/LeWagonIcon.png" width="150" style="margin-bottom: 10px;" />
                <div>Created by Le Wagon Data Science Batch #2012</div>
                <div style="font-style: italic; margin-top: 8px;">
                    Built with ❤️ by the SolarSoundBytes team
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
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
