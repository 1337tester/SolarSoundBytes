import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

def articles_api_2_csv(t_start: str, t_end: str):
    """
    Fetches articles from GNews API for the specified time window
    and saves them to a CSV file labeled with the time range.

    Parameters:
    - t_start (str): ISO 8601 start timestamp, e.g., '2022-01-02T00:00:00Z'
    - t_end (str): ISO 8601 end timestamp, e.g., '2022-07-02T23:59:59Z'
    """

    # load API key from .env file
    load_dotenv()
    API_KEY = os.getenv("GNEWS_API_KEY")

    if not API_KEY:
        raise ValueError("GNEWS_API_KEY not found in environment variables.")

    # define GNews API endpoint and parameters
    url = 'https://gnews.io/api/v4/search'
    params = {
        'q': '"Renewable Energy" AND "Energy Storage"',
        'lang': 'en',
        'max': 25,
        'apikey': API_KEY,
        'from': t_start,
        'to': t_end,
        'expand': 'content'
    }

    # API call
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API call failed with status {response.status_code}: {response.text}")

    data = response.json()

    if 'articles' not in data or not data['articles']:
        print(f"No articles found between {t_start} and {t_end}")
        return

    # convert data to DataFrame
    df = pd.DataFrame(data['articles'])

    # compose filename
    date_fmt = "%Y-%m-%d"
    start_date_str = datetime.fromisoformat(t_start.replace("Z", "")).strftime(date_fmt)
    end_date_str = datetime.fromisoformat(t_end.replace("Z", "")).strftime(date_fmt)
    filename = f"articles-from-{start_date_str}-to-{end_date_str}.csv"

    # define output path
    output_folder = '../data/csv/'

    # throw error if output folder does not exist
    if not os.path.isdir(output_folder):
        print(f"Output folder '{output_folder}' does not exist. No file saved.")
        return

    # add filename to output path if it exists and save CSV
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} articles to {output_path}")









'''
call articles_api_2_csv() in 6-Month Intervals
'''

from datetime import datetime
from dateutil.relativedelta import relativedelta

def run_articles_api_in_month_chunks(t_total_start, t_total_end, window_months=6):

    t_start = t_total_start

    while t_start < t_total_end:
        # Add 6 months to t_start to get t_end
        t_end = t_start + relativedelta(months=window_months) - relativedelta(seconds=1)

        # Make sure we don't overshoot the total end time
        if t_end > t_total_end:
            t_end = t_total_end

        # Format datetime to ISO 8601 string
        t_start_str = t_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        t_end_str = t_end.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Call the function
        articles_api_2_csv(t_start_str, t_end_str)

        # Move to the next 6-month chunk
        t_start = t_end + relativedelta(seconds=1)




if __name__ == "__main__":

    # Define total time period
    t_total_start = datetime.fromisoformat('2022-01-02T00:00:00'.replace("Z", ""))
    t_total_end = datetime.fromisoformat('2024-12-24T23:59:59'.replace("Z", ""))

    # t_total_start = '2022-01-02T00:00:00Z'
    # t_total_end = '2025-01-02T23:59:59Z'

    # loop through entire time period, using window size of 6 months each
    run_articles_api_in_month_chunks(t_total_start, t_total_end, 6)

    # t_start = '2022-01-02T00:00:00Z'
    # t_end = '2022-07-02T23:59:59Z'
    # articles_api_2_csv(t_start, t_end)
