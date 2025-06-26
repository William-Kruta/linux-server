import os
import json


FILE_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(FILE_DIR, "config.json")


def read_config():
    with open(FILE_PATH, "r") as f:
        data = json.load(f)
    return data


def get_server():
    data = read_config()
    return data["server"]


def get_time_zone():
    data = read_config()
    return data["timezone"]
