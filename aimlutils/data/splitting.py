from sklearn.model_selection import train_test_split as _train_test_split
from typing import List
import pandas as pd
import numpy as np

# To do: Add documentation, a logger and verbose options

def stratified_split(df: pd.DataFrame,
                     frac: float,
                     column: List[str]) -> (pd.DataFrame, pd.DataFrame):
    
    label_count = df[column].value_counts().to_dict()
    labels_we_can_use = df[column].apply(lambda x: label_count[x] > 1)
    items_with_count_one = df[~labels_we_can_use].copy()
    items_needing_split = df[labels_we_can_use].copy()
    
    train, test = _train_test_split(
        items_needing_split,
        test_size=frac,
        stratify=items_needing_split[column]
    )
    train = pd.concat([train, items_with_count_one], axis = 0, sort = True)#.reset_index(drop = True)
    return train, test


def train_test_split(df: pd.DataFrame,
                     fraction: float = 0.2) -> (pd.DataFrame, pd.DataFrame):
    
    fraction = min(1.0, fraction)
    train, test = stratified_split(df, fraction, "label")     
    return train, test


def train_test_val_split(df: pd.DataFrame,
                         fraction: float = 0.2) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    
    fraction = min(1.0, fraction)
    train, _test = stratified_split(df, fraction, "label") 
    test, val = stratified_split(_test, 0.5, "label") 
    
    return train, test, val