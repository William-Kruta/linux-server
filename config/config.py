import os
import json

FILE_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(FILE_DIR, "config.json")


def read_config() -> dict:
    with open(FILE_PATH, "r") as f:
        file = json.load(f)
    return file


def get_candles_path() -> str:
    file = read_config()
    path = file["database"]["candles"]["path"]
    return path


def get_options_path() -> str:
    file = read_config()
    path = file["database"]["options"]["path"]
    return path


def get_response_model() -> str:
    file = read_config()
    model = file["llm"]["response_model"]
    return model
