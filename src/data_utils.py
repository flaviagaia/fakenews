from pathlib import Path

import pandas as pd


TEXT_COLUMNS = ["title", "content", "tags"]


def load_csv(path):
    return pd.read_csv(path).fillna("")


def combine_text_columns(frame):
    combined = frame.copy()
    combined["text"] = combined[TEXT_COLUMNS].agg(" ".join, axis=1)
    return combined


def data_path(*parts):
    return Path(__file__).resolve().parents[1] / "data" / Path(*parts)
