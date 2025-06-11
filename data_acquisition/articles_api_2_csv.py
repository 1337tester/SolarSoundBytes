import os
import re
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


def articles_api_2_csv(t_start_str: str, t_end_str: str, query: str, query_subdivisions: int = 1):
    """
    Fetches articles from GNews API for the specified time window
    and saves them to CSV file(s) labeled with query and time range(s).

    Parameters:
    - query_example = '"Renewable Energy" OR "Energy Storage"'
    - t_start_str: ISO 8601 start timestamp, e.g., '2022-01-02T00:00:00Z'
    - t_end_str  : ISO 8601  end  timestamp, e.g., '2022-07-02T23:59:59Z'
    """

    # --------------------- API call ---------------------

    # load API key and plan-specific max_n_articles from .env file
    load_dotenv()
    API_KEY = os.getenv("GNEWS_API_KEY")
    MAX_N_ARTICLES = os.getenv("GNEWS_MAX_N_ARTICLES")

    if not API_KEY:
        raise ValueError("GNEWS_API_KEY not found in environment variables.")

    # define GNews API endpoint and parameters
    url = 'https://gnews.io/api/v4/search'
    params = {
        'q': query,
        'lang': 'en',
        'max': 25,           # max for essential plan is 25 articles per request
        'apikey': API_KEY,
        'from': t_start_str,
        'to': t_end_str,
        'expand': 'content'
    }

    # call API and store data if successful
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API call failed with status {response.status_code}: {response.text}")
    data = response.json()

    # add timeout between API calls to prevent being locked out
    '''
    Essential plan as per GNews API pricing = https://gnews.io/#pricing
    - 1000 requests per day
    - Up to 25 articles returned per request
    - Maximum of 4 requests per 1 second
        --> t_min between requests = 1/4 = 0.25 seconds
    - Full article content and pagination
    '''
    time.sleep(0.5)  # double the minimum of 1/4 = 0.25 seconds to be on the safe side

    if 'articles' not in data or not data['articles']:
        print(f"No articles found between {t_start_str} and {t_end_str}")
        return

    # convert data to DataFrame
    df = pd.DataFrame(data['articles'])


    # -------------------- check if max_n_articles reached --------------------

    '''
    call API with specific query from queries list and target time period
    if max number of articles reached
    - half time_window_size
    - store updated start time (t_start_half_window is now closer to t_end)
    - run again using (from old t_start to t_end_add_run = t_start_updated)

    - recursively subdividing until no sub-window is overfilled or the window is too small.
    - For a very dense period (e.g., news spike), it’ll keep splitting until it can fit all articles.
    '''

    from datetime import timedelta

    if len(df) >= int(MAX_N_ARTICLES):
        print(f"Query \n'{query}'\n reached maximum number of {MAX_N_ARTICLES} articles.")
        print("To ensure no articles are missed, divide query into two half time ranges, \n\
            by moving t_start towards fixed t_end and call API again using \n\
            {query_subdivisions} query_subdivisions.")
        print(f"Original time range: {t_start_str} to {t_end_str}")

        # Parse start and end times
        t_start = datetime.fromisoformat(t_start_str.replace("Z", ""))
        t_end = datetime.fromisoformat(t_end_str.replace("Z", ""))
        time_span = (t_end - t_start).total_seconds()

        # Don't subdivide if the window is less than, say, 2 days (avoid infinite splits)
        if time_span < 2 * 24 * 3600:
            print("Time window too small to subdivide further.")
            return

        # Calculate mid-point and create two sub-windows
        mid = t_start + (t_end - t_start) / 2
        mid = mid.replace(microsecond=0)  # Optional: clean microseconds

        # Recursively call for each half
        articles_api_2_csv(
            t_start_str=t_start_str,
            t_end_str=mid.strftime('%Y-%m-%dT%H:%M:%SZ'),
            query=query,
            query_subdivisions=query_subdivisions + 1
        )
        articles_api_2_csv(
            t_start_str=(mid + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            t_end_str=t_end_str,
            query=query,
            query_subdivisions=query_subdivisions + 1
        )

        # print recursion depth
        print(f"Recursion depth: {query_subdivisions + 1}")

        return  # Prevent saving/output for the overfull window






    # ---------------------------- compose filename ----------------------------
    '''
    prepare query_string from query following below rules and examples

    rules:
    - convert all to lowercase EXCEPT logical operators (AND, OR, NOT)
    - join words with hyphens (-)
    - remove double quotes (") and only leave single surrounding quotes (')

    examples:
    - example_query = '"Renewable Energy" OR "Energy Storage"'
    - example_query_string = 'renewable-energy-OR-energy-storage'
    '''

    # Remove double quotes
    query_no_quotes = query.replace('"', '')

    # Split by logical operators (AND, OR, NOT) while keeping them
    tokens = re.split(r'\b(AND|OR|NOT)\b', query_no_quotes)

    # Process each token
    processed_tokens = []
    for token in tokens:
        token_strip = token.strip()
        if token_strip in {"AND", "OR", "NOT"}:
            processed_tokens.append(token_strip)
        elif token_strip:
            # Lowercase and replace spaces with hyphens
            processed_tokens.append(token_strip.lower().replace(' ', '-'))

    # Join all tokens with hyphens
    query_string = '-'.join(processed_tokens)

    # count number of articles
    number_of_articles = len(df)

    # format start and end dates for filename, add optional query_subdivisions to output filename
    date_fmt = "%Y-%m-%d"
    start_date_str = datetime.fromisoformat(t_start_str.replace("Z", "")).strftime(date_fmt)
    end_date_str = datetime.fromisoformat(t_end_str.replace("Z", "")).strftime(date_fmt)
    filename = f"gnews-query-{query_string}-yields-{number_of_articles}-articles-from-{start_date_str}-to-{end_date_str}.csv"

    # define output path
    output_folder = '../data/csv/gnews_articles/'

    # throw error if output folder does not exist
    if not os.path.isdir(output_folder):
        print(f"Output folder '{output_folder}' does not exist. No file saved.")
        return

    # add filename to output path if it exists and save CSV
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} articles to {output_path}")


# ------------------ loop through API calls in main function -------------------


if __name__ == "__main__":

    t_start_str = '2022-01-02T00:00:00Z'
    t_end_str = '2024-12-24T23:59:59Z'

    query = '"Renewable Energy" OR "Energy Storage"'

    articles_api_2_csv(t_start_str, t_end_str, query, query_subdivisions=1)
