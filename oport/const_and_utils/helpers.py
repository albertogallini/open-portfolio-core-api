# helpers.py
# General-purpose helper utilities.

import pandas as pd


def get_ticker(input_file: str) -> str:
    import re

    pattern = r"_([^_]*)\."
    ticker = re.findall(pattern, input_file)
    return ticker


def append_today_date(input_folder_name: str) -> str:
    from datetime import date

    today = date.today()
    today_str = today.strftime("%d-%m-%Y")
    output_folder_name = input_folder_name + "/" + today_str + "/"
    return output_folder_name


def create_instance_of_class(class_type, *args, **kwargs) -> object:
    return class_type(*args, **kwargs)


def to_serializable(data):
    if isinstance(data, dict):
        return {k: to_serializable(v) for k, v in data.items()}
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_dict()
    return data
