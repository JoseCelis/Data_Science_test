import os
import sys
import logging
from dotenv import load_dotenv, find_dotenv
from src import process_data, pre_model, model

load_dotenv(find_dotenv())


def main():
    process_data.main()
    model.main()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
