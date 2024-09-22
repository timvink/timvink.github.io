"""
Run using:

```bash
uv run --python 3.10 benchmark.py
# or
for py in 3.10 3.11 3.12; do uv run --quiet --python-preference only-managed --python $py benchmark.py; done
``` 

"""
import sys
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from functools import lru_cache
from memo import memfile, time_taken

@lru_cache
def load_dataset(dataset_name:str) -> tuple[pd.DataFrame, pd.Series]:
    if dataset_name == "adult":
        return fetch_openml("adult", version=2, return_X_y=True, cache=True)
    if dataset_name == "covertype":
        return fetch_openml("covertype", version=3, return_X_y=True, cache=True)
    if dataset_name == "Click_prediction_small":
        return fetch_openml("Click_prediction_small", version=1, return_X_y=True, cache=True)
    return fetch_openml(dataset_name, return_X_y=True, cache=True)


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    hist = HistGradientBoostingClassifier(categorical_features="from_dtype")

    hist.fit(X_train, y_train)
    y_decision = hist.decision_function(X_test)
    _ = roc_auc_score(y_test, y_decision)
    return X_train.shape


@memfile(filepath="output/results.json")
@time_taken()
def experiment(dataset_name):
    X, y = load_dataset(dataset_name)
    n_rows, n_cols = train(X, y)
    # Size of X in memory ( in MB)
    size = X.memory_usage().sum() / 1024 ** 2
    return {"python": sys.version, "n_rows": n_rows, "n_cols": n_cols, "size_mb": size}


if __name__ == "__main__":
    n_repeats = 20
    datasets = ["adult", "click_prediction_small"]
    # Warm up the cache
    for dataset in datasets:
        # Warm up cache
        _ = load_dataset(dataset)
        print(f"Starting dataset '{dataset}'")
        for _ in tqdm(range(n_repeats)):
            experiment(dataset_name=dataset)




