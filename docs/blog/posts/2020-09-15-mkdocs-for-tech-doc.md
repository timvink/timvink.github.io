---
date: 2020-09-15 7:00:00
slug: mkdocs-for-tech-doc
---

# Using MkDocs for technical reporting

In this post I will explain why [MkDocs](https://www.mkdocs.org/) is well suited for writing technical reports on machine learning models and introduce relevant parts of the MkDocs ecosystem.

<!-- more -->

## Introduction

Machine learning might be a sexy field, writing documentation for models is not. [towardsdatascience.com](https://towardsdatascience.com) hardly has any blogposts [on writing documentation](https://towardsdatascience.com/search?q=documentation&ref=opensearch). Spending time on writing good documentation with your models is often overlooked.

In [my work](../../about.md) writing good technical documentation is an essential part of a machine learning project. The docs contain business context, list model performance metrics, document procedures around model updating and monitoring, explain important decisions, capture many crucially important considerations around model fairness, ethics, explainability, model limitations and much more. Our model documentation template contains guidelines that help ensure quality and consistency between projects, and promote best practices.

Writing documentation cannot be made more fun, but it can be made easier. There a lot of different tools and packages out there, and it can take a long time to find a good setup to create a smooth documentation workflow. In this post I'll discuss some alternatives and introduce you to MkDocs and some of the most useful tools in the MkDocs ecosystem for writing technical documentation.

## MkDocs vs alternatives

A good solution for writing documentation should be:

- Version controllable (flat files part of your git repo)
- Easy to convert to HTML and PDF
- Easy to learn  (also for less-technical project members)
- Easy to edit (also for less-technical project members)
- Able to insert content dynamically
- Python-based (easy to install)
- Really good looking! 💃

[MkDocs](https://www.mkdocs.org/) fits all these requirements, for but reference, there are some alternatives:

- There are several solutions that use [nbconvert](https://nbconvert.readthedocs.io/) to convert jupyter notebooks to HTML or PDF. [fastpages](https://github.com/fastai/fastpages) creates a blog site from a directory of notebooks, and the newly rewritten [jupyter-book](https://medium.com/swlh/the-new-jupyter-book-4028f054893f) is a great solution for writing an entire book from notebook(s). The [notebook file format](https://nbformat.readthedocs.io/en/latest/) however is JSON-based and contains not only markdown, but also the code and the output. That makes it hard to collaborate on a notebook in a team through version control (f.e. reviewing changes).
- Then there is [Sphinx](https://www.sphinx-doc.org/en/master/), a mature package initially for documenting python packages. It has a large eco-system of packages to extend sphinx, and can also support markdown via [recommonmark](https://www.sphinx-doc.org/en/1.6/markdown.html). However I find it is harder to setup and learn if you're just looking to write simple documentation instead of documenting python objects.

## MkDocs

[MkDocs](https://www.mkdocs.org/) is a simple site generator that can create a website from a directory of markdown files. It's easy to extend through 1) themes, 2) plugins and 3) extensions to the markdown language. Because it uses simple markdown files, you can write docs in your IDE and use your normal git workflow. That keeps your code in sync with your docs, which definitely beats sending around lots of `report_final22.docx` files via email!

Because MkDocs offers such a large eco-system, it's easy to miss some of the best functionalities. The rest of the blog will give you a quick tour:

### The bare basics

In your project directory, run:

```bash
pip install mkdocs
mkdocs new .
```

Which will create these files:

```bash
./
├── docs
│   └── index.md
└── mkdocs.yml
```

Which you view using `mkdocs build` (creates a new `site/` directory with your website), or using `mkdocs serve` (starts a local webserver).

The `mkdocs.yml` file is where you can customize and extend MkDocs. One setting I recommend changing is setting [use_directory_urls](https://www.mkdocs.org/user-guide/configuration/#use_directory_urls) to `false`.
This ensures you can specify relative paths to images and your website navigation works using local files as well.

```yml
# mkdocs.yml
use_directory_urls : False
```

### Adding a theme

There are several [themes for mkdocs](https://github.com/mkdocs/mkdocs/wiki/MkDocs-Themes), but there is only one really killer theme for MkDocs: [mkdocs-material](https://github.com/squidfunk/mkdocs-material). To enable it, `pip install mkdocs-material` and add these lines to your `mkdocs.yml`:

```yml
# mkdocs.yml
theme:
  name: material
```

The [mkdocs-material theme documentation](https://squidfunk.github.io/mkdocs-material/) is very structured and offers many options for customizing the theme (including a dark mode!).

### Adding markdown extensions

The basic markdown syntax can be expanded to include other use cases. The [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) theme already has styling support for the extensions in the [pymdown-extensions](https://facelessuser.github.io/pymdown-extensions/) package and has examples and guides for many of them. Make sure to consider enabling these extensions when writing technical reports:

- [Abbreviations](https://squidfunk.github.io/mkdocs-material/reference/abbreviations/): Which also allow for defining a glossary centrally
- [Admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/): (collapsible) content blocks
- [Code highlighting](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/): Add code blocks with highlighting
- [Data tables](https://squidfunk.github.io/mkdocs-material/reference/data-tables/): Well styled tables
- [footnotes](https://squidfunk.github.io/mkdocs-material/reference/footnotes/): Add footnotes for references
- [content tabs](https://squidfunk.github.io/mkdocs-material/reference/content-tabs/): Separate content into tabs
- [MathJax](https://squidfunk.github.io/mkdocs-material/reference/mathjax/): Add formulas to your page
- [emojis](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/): 😎

For reference, there are many [more packages](https://github.com/Python-Markdown/markdown/wiki/Third-Party-Extensions) that extend the markdown syntax and can be used with MkDocs.

### Adding plugins

MkDocs is python based, and allows you to write plugins that execute scripts during many different points in the build process (where markdown is converted to HTML).
There are [a lot](https://github.com/mkdocs/mkdocs/wiki/MkDocs-Plugins) of useful plugins. I'd like to highlight five which I wrote specifically to make writing reproducible technical documents easier:

- [mkdocs-print-site-plugin](https://github.com/timvink/mkdocs-print-site-plugin): Let users save your entire site as a PDF through *File > Print > Save as PDF*.
- [mkdocs-git-authors-plugin](https://github.com/timvink/mkdocs-git-authors-plugin): Display authors (from git) at the bottom of each page.
- [mkdocs-git-revision-date-localized-plugin](https://github.com/timvink/mkdocs-git-revision-date-localized-plugin): Adds a 'Last updated' date (from git) to bottom of each page.
- [mkdocs-table-reader-plugin](https://github.com/timvink/mkdocs-table-reader-plugin): Directly insert CSV's using a `{% raw %}{{ read_csv('table.csv') }}{% endraw %}` syntax.
- [mkdocs-enumerate-headings-plugin](https://github.com/timvink/mkdocs-enumerate-headings-plugin): Enumerate all headings across your website in order.

Some other very useful plugins to explore:

- [mknotebooks](https://github.com/greenape/mknotebooks): Add your jupyter notebooks directly to your MkDocs site.
- [mkdocs-awesome-pages-plugin](https://github.com/lukasgeiter/mkdocs-awesome-pages-plugin): Simplifies configuring page titles and their order
- [mkdocs-pdf-export-plugin](https://github.com/zhaoterryy/mkdocs-pdf-export-plugin): Programmatically export to PDF
- [mkdocs-bibtex](https://github.com/shyamd/mkdocs-bibtex): Citation management using bibtex
- [mkdocs-minify-plugin](https://github.com/byrnereese/mkdocs-minify-plugin): Minifies HTML and/or JS files prior to being written to disk (faster page loading)

## Conclusion & further reading

MkDocs enables writing elegant docs that live next to your source code, and is a great fit for writing technical reports in data science teams. With the theme, plugins and markdown extensions introduced, you should have a great place to get started. For more info, see:

- [mkdocs.org](https://www.mkdocs.org/)
- [mkdocs-material](https://squidfunk.github.io/mkdocs-material/)
- [list of mkdocs all plugins](https://github.com/mkdocs/mkdocs/wiki/MkDocs-Plugins)
- [calmcode.io intro to mkdocs](https://calmcode.io/mkdocs/intro-to-mkdocs.html)
