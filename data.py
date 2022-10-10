import config 
from utils import download_data, process_cerist2022_data
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", default=False, help="Must be True or False", action='store_true')
    parser.add_argument("--process", default=False, help="Must be True or False", action='store_true')
    args = parser.parse_args()

    if args.download:
        download_data(config.DATA_PATH, config.DATA_URL)
    if args.process:
        process_cerist2022_data(config.DATA_PATH, config.DATASET_TEXT)


