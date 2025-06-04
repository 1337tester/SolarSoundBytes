'''
- Load large CSV file (climate_change_twitter.csv > 2GB)
- use Dask to efficiently handle large file by reading CSV in chunks
- filter tweets inside UK boundaries

only rows with (lng, lat) coordinates inside the UK
Ignores rows with missing or out-of-bounds coordinates
Saves the result to a new CSV file uk_tweets.csv

- save the filtered tweets to a new CSV file
'''

import geopandas as gpd
from shapely.geometry import Point
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os
import multiprocessing

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data')

# Official UK (=GBR) boundary shapefile gadm41_GBR_0.shp obtained from GADM:
# https://gadm.org/download_country.html
# Use level 0 for country boundary
UK_SHP_FILE = os.path.join(DATA_DIR, 'shp/gadm41_GBR_0.shp')
INPUT_CSV = os.path.join(DATA_DIR, 'csv/climate_change_twitter.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'csv/uk_tweets.csv')

def load_uk_boundary():
    try:
        uk_gdf = gpd.read_file(UK_SHP_FILE)
        return uk_gdf.unary_union  # Combine all polygons into one
    except Exception as e:
        print(f"Error loading UK boundary data: {e}")
        print(f"Make sure {UK_SHP_FILE} exists")
        return None

def is_in_uk(lng, lat, uk_polygon):
    """Check if a point is within UK boundaries"""
    if lng is None or lat is None:
        return False
    try:
        return uk_polygon.contains(Point(float(lng), float(lat)))
    except:
        return False

def process_tweets():
    """Process the climate change tweets dataset"""
    # Load UK boundary
    uk_polygon = load_uk_boundary()
    if uk_polygon is None:
        return

    # Calculate optimal chunk size and number of workers
    n_cores = multiprocessing.cpu_count()
    chunk_size = "256MB"  # Increased chunk size for better performance

    try:
        print("Reading CSV file...")
        df = dd.read_csv(
            INPUT_CSV,
            usecols=['id', 'lng', 'lat'],
            blocksize=chunk_size,
            assume_missing=True,  # Faster CSV parsing
            dtype={
                'id': 'str',
                'lng': 'float64',
                'lat': 'float64'
            }
        )

        print(f"Number of partitions: {df.npartitions}")
        print(f"Using {n_cores} CPU cores")

        # Pre-filter obvious non-UK coordinates to reduce processing
        uk_bounds = {
            'lat_min': 49.9,  # Southernmost point of UK
            'lat_max': 60.9,  # Northernmost point of UK
            'lng_min': -8.6,  # Westernmost point of UK
            'lng_max': 1.8    # Easternmost point of UK
        }

        df = df[
            (df.lat >= uk_bounds['lat_min']) &
            (df.lat <= uk_bounds['lat_max']) &
            (df.lng >= uk_bounds['lng_min']) &
            (df.lng <= uk_bounds['lng_max'])
        ]

        # Filter tweets within UK using optimized computation
        print("Filtering tweets within UK boundaries...")
        df['in_uk'] = df.map_partitions(
            lambda partition: partition.apply(
                lambda row: is_in_uk(row['lng'], row['lat'], uk_polygon),
                axis=1
            ),
            meta=('in_uk', 'bool')
        )

        # Use ProgressBar to show computation progress
        with ProgressBar():
            print("Computing results...")
            uk_tweets = df[df['in_uk']].compute(
                scheduler='processes',
                num_workers=n_cores
            )

        print(f"Found {len(uk_tweets)} tweets in UK")

        # Save filtered tweets efficiently
        print("Saving filtered tweets...")
        uk_tweets.drop(columns=['in_uk']).to_csv(
            OUTPUT_CSV,
            index=False,
            chunksize=10000  # Efficient writing
        )
        print(f"Saved {len(uk_tweets)} UK tweets to {OUTPUT_CSV}")

    except Exception as e:
        print(f"Error processing tweets: {e}")

if __name__ == "__main__":
    process_tweets()
