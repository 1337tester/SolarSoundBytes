import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
# api_key = os.getenv("API_KEY")

api_key = os.getenv("API_KEY")


def create_text_from_sent_analy_df(data_twitter, data_news, data_sp500, data_energy):

    prompt = f"""
    Here you have datasets regarding sentiment analysis of tweets, newsarticles
    the s&p 500 and the installed capacity of solar and wind plants over time
    {data_twitter}
    {data_news}
    {data_sp500}
    {data_energy}

    Please summarize the development of public opinion
    - Explain whether the perception in social media and in news media was different
    and why based on the events below. Feel free to add your own knowledge
    - Events:
        -  2022-02-24 Russias invasion of Ukraine & global energy crisis
        - 2022-05-18 EU adopts REPowerEU plan (cut Russian fuel, boost renewables)
        - 2022-08-16 US Inflation Reduction Act
        - 2023-05-30 Solar > Oil investment tipping point (IEA Report)
        - 2023-12-12 COP28 climate summit - pledge to triple renewables by 2030
        - 2022-12-31 Global solar capacity surpasses 1 TW (year-end milestone)

    Write a structured text of about 100 words.

    """

    client = OpenAI(api_key = api_key)
    response = client.chat.completions.create(
        model="gpt-4o",  #
        messages=[
            {"role": "system", "content": "You are a data-analytical journalist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    # 🖨️ Ausgabe
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

# output = create_text_from_sent_analy_df(result_twitter, result_news)
# print(output)
