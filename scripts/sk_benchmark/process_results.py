"""
Run using

```bash
uv run process_results.py
```
"""

import json
import pandas as pd

df = pd.read_json("output/results.json", lines=True)

df['python'] = 'python ' + df['python'].str.split(' ', expand=True)[0]


def prepare_vegalite_boxplot_data(df):
    # Group by python version and calculate the required statistics
    grouped = df.groupby('python')['time_taken'].agg([
        ('lower', lambda x: x.quantile(0.00)),
        ('q1', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('q3', lambda x: x.quantile(0.75)),
        ('upper', lambda x: x.quantile(1.00))
    ]).reset_index()

    # Convert to the desired JSON format
    return grouped.to_dict('records')


for name in df['dataset_name'].unique():
    with open(f'output/benchmark_python_{name}.json', 'w') as f:
        json_data = prepare_vegalite_boxplot_data(df.query(f"dataset_name == '{name}'"))
        json.dump(json_data, f, indent=4)
