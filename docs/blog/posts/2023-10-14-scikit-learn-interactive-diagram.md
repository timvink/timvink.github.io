---
date: 2023-10-14 8:00:00
slug: sklearn-visualizations-in-mkdocs
authors:
  - timvink
tags:
  - today-I-learned
---

# Inserting interactive scikit-learn diagrams into mkdocs

`scikit-learn` has this nice feature where you can [display an interactive visualization of a pipeline](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_pipeline_display.html). 
This post shows how to insert interactive diagrams into your mkdocs documentation, which is great for documenting your machine learning projects.

<!-- more -->

Here's an example of what it looks like [^1]:

--8<-- "docs/assets/visualizations/gridsearch.html"

## How it's done

To insert a pipeline visualization into a markdown document, first save the `.html` file:

```python
from sklearn.utils import estimator_html_repr

with open("docs/assets/visualizations/gridsearch.html", "w") as f:
    f.write(estimator_html_repr(grid_search))
```

Then, insert it into mkdocs using the [snippets extension](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/), see [embedding external files](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/#embedding-external-files):

```markdown
;--8<-- "docs/assets/visualizations/gridsearch.html"
```

Alternatively, you could use the [markdown-exec](https://github.com/pawamoy/markdown-exec) package, or a [mkdocs hook](https://www.mkdocs.org/user-guide/configuration/#hooks) with a python script that is triggered when the docs are built (`on_build` event).

[^1]: the `grid_search` pipeline is from [this example](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_pipeline_display.html#displaying-a-grid-search-over-a-pipeline-with-a-classifier)

