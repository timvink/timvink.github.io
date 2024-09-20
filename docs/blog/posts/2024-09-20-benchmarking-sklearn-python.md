---
date: 2024-09-20 23:00:00
slug: benchmarking-sklearn-python
authors:
  - timvink
---

# Benchmarking scikit-learn across python versions using `uv`

When python 3.11 came out 2 years ago (24 October 2022) it promised to be 10-60% faster than python 3.10, and 1.25x faster on the standard benchmark suite (see the [what's new in 3.11](https://docs.python.org/3/whatsnew/3.11.html)). I've always wondered how that translates to training machine learning models in python, but I couldn't be bothered to write a benchmark. That is, until [astral](https://astral.sh/) released [uv 0.4.0](https://github.com/astral-sh/uv/releases/tag/0.4.0) which introduces ["_a new, unified toolchain that takes the complexity out of Python development_"](https://astral.sh/blog/uv-unified-python-packaging).

`uv` has been blowing my mind and is transforming the way I work, and there are many resources out there already discussing it (like [Rye and uv: August is Harvest Season for Python Packaging](https://lucumr.pocoo.org/2024/8/21/harvest-season/) and [uv under discussion on Mastodon](https://simonwillison.net/2024/Sep/8/uv-under-discussion-on-mastodon/)). One of the new capabilities is that `uv python` can _bootstrap and install Python for you_. Instead of building python from source, `uv` uses (and contributes to) the [python standalone builds](https://gregoryszorc.com/docs/python-build-standalone/main/) project. For each python version they will pre-build python binaries suitable for a wide range of system architectures (currently 773 builds per python version).

The CEO of Astral (creator of `uv`) is Charlie Marsh, and he recently appeared on the _Talk Python To Me_ podcast ([Episode #476 unified packaging with uv](https://talkpython.fm/episodes/show/476/unified-python-packaging-with-uv)). There he explained that these python builds _"will be noticeably faster than what you would get by default with [PyEnv](https://github.com/pyenv/pyenv)"_ because they are compiled with optimizations. And because it's a standalone binary, the installation speed is restricted to the time it takes to stream and unzip it down into disk. It now takes me ~10-20 _seconds_ to install a new python version!

## The benchmark

We'll take a scikit-learn example for the `HistGradientBoostingClassifier` using the [adult](https://www.openml.org/search?type=data&sort=runs&id=1590&status=active) openml dataset (binary classification with 49k rows). 

To setup the benchmark project:

```bash
uv init sk_benchmark && cd sk_benchmark
uv add scikitlearn pandas
```

Then we can add a `benchmark.py` script:

```python
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
import sys

X_adult, y_adult = fetch_openml("adult", version=2, return_X_y=True)

def train(X_adult, y_adult):
    X_adult = X_adult.drop(["education-num", "fnlwgt"], axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(X_adult, y_adult, random_state=0)
    hist = HistGradientBoostingClassifier(categorical_features="from_dtype")

    hist.fit(X_train, y_train)
    y_decision = hist.decision_function(X_test)
    auc = roc_auc_score(y_test, y_decision)

train_times = []
for _ in range(10):
    tik = time.time()
    train(X_adult, y_adult)
    train_times.append(time.time() - tik)

print(f"Python {sys.version}.")
print(f"Average time: {sum(train_times) / len(train_times):.4f}s")
```

Then this is the real party trick:

```bash
for py in 3.10 3.11 3.12; do uv run --quiet --python $py benchmark.py; done
``` 

## The results

```
Python 3.10.13 (main, Jul 11 2024, 16:23:02) [GCC 9.4.0].
Average time: 0.9763s
Python 3.11.10 (main, Sep  9 2024, 22:11:19) [Clang 18.1.8 ].
Average time: 0.9430s
Python 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0].
Average time: 0.9507s
```

So python 3.11 is ~4% faster than 3.10, and.. wait what? python 3.12 _slower_ ?!

You might have noticed the python build source is different. Luckily, `uv` has a `--python-preference` option which you can set to `managed-only`. This means we only get the optimized builds from the python-standalone-builds project. This gives us:

```
Python 3.10.15 (main, Sep  9 2024, 22:15:21) [Clang 18.1.8 ].
Average time: 0.9362s
Python 3.11.10 (main, Sep  9 2024, 22:11:19) [Clang 18.1.8 ].
Average time: 0.9159s
Python 3.12.6 (main, Sep  9 2024, 22:11:19) [Clang 18.1.8 ].
Average time: 0.9573s
```

Of course there is still some variability, but the differences are not _that_ noticable.
Digging a bit deeper, it turns out python 3.12 is indeed slower than python 3.11. Of course it depends (see this [extensive comparison benchmark](https://en.lewoniewski.info/2023/python-3-11-vs-python-3-12-performance-testing/)).

## Conclusion

This little benchmark seems to show the performance difference between python versions for training a basic machine learning classifier is not that big of a deal.

Another idea to try is to switch from `pandas` to [`polars`](https://github.com/pola-rs/polars) dataframes, which `scikit-learn` supports since January 2024 ([scikit-learn 1.4+](https://scikit-learn.org/dev/whats_new/v1.4.html#version-1-4-0)).
Queries using [polars](https://github.com/pola-rs/polars) dataframes are 10-100x faster than [pandas](https://github.com/pandas-dev/pandas) dataframes ([benchmark](https://pola.rs/posts/benchmarks/)). On top of that, polars just [released a new accelerated GPU engine with nvidia](https://pola.rs/posts/gpu-engine-release/) that promises another 2-13x speedup. Of course all that doesn't translate to training machine learning models directly.. a first try didn't show any significant training time improvements. As always, it depends..
