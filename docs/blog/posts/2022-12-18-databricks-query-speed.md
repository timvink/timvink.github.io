---
date: 2022-12-19 8:00:00
slug: databricks-query-speed
---

# Speeding up databricks SQL queries

Retrieving data from a datawarehouse is a common operation for any data scientist. In August 2021 databricks released a blog post describing [how [Databricks] achieved high-bandwidth connectivity with BI-tools](https://www.databricks.com/blog/2021/08/11/how-we-achieved-high-bandwidth-connectivity-with-bi-tools.html). In it, they introduced *cloud fetch,* promising a 12x experimental speedup on a dataset with 4M rows and 20 columns, achieved mainly by doing downloads in parallel. When I read this I immediately dove head-first into the rabbit hole, hoping to reduce the time from running a SQL query to having it inside a `pandas` dataframe. This blogpost details the journey on how I achieved a significant speedup for our databricks queries.

<!-- more -->

**Update 2023**: Since the `databricks-sql-connector` [v2.8.0 release](https://github.com/databricks/databricks-sql-python/releases/tag/v2.8.0) from July 2023, there is support for cloudfetch. Enabling this leads to the fasted method, and I have updated the post benchmark and example code below.

## The baseline

Our reference dataset is a sample of 2M rows from a table with 146 columns of mixed types. Initially, I was using a basic setup of [databricks SQL connector](https://docs.databricks.com/dev-tools/python-sql-connector.html). Very easy setup and it worked great for smaller queries, but for larger queries it got slow quickly:

| Method | Speed |
| --- | --- |
| Baseline | 6m57s |

I had already figured out one optimization while browsing the documentation: using `.fetchall_arrow()` ([link](https://docs.databricks.com/dev-tools/python-sql-connector.html#fetchall_arrow-method)) instead of `.fetchall()`. This “gets all (or all remaining) rows of a query, as a PyArrow table”. That helped a lot:

| Method | Speed |
| --- | --- |
| Baseline | 6m57s |
| .fetchall_arrow | 3m38s |

## Cloudfetch + Simba ODBC drivers

Hoping for blazing speeds, I set up the databricks custom ‘Simba’ ODBC drivers [as instructed](https://docs.databricks.com/integrations/jdbc-odbc-bi.html). Getting the connection string exactly right together with active directory tokens took quite an effort, but once I got connected I ran the benchmark:

| Method | Speed |
| --- | --- |
| Baseline | 6m57s |
| .fetchall_arrow | 3m38s |
| Cloudfetch | 4m24s |

Significantly *slower !* This was disappointing. I re-ran the benchmark in different time periods but busy clusters could not explain the slower results. I had to dig deeper.

Reading everything I could find online about cloud fetch and the databricks ODBC drivers, it seems you cannot see whether Cloudfetch is actually enabled or working (update: you can now [explicitly set the cloud fetch override](https://docs.databricks.com/integrations/jdbc-odbc-bi.html#set-the-cloud-fetch-override) on a cluster configuration). ****I did [find a section](https://docs.databricks.com/integrations/bi/jdbc-odbc-bi.html#advanced-configurations) stating Databricks automatically disables Cloud Fetch for S3 buckets that have enabled versioning. I checked with an infra engineer, and this was not the case. We tried running queries over a custom proxy to monitor traffic, and it did seem multiple connections were opened. ODBC logs showed the file connections also.

So, likely Cloudfetch was working, but something else was going on.

## Back to basics

I estimated the final `pandas` dataset size using pandas’s `.memory_usage(deep=True)` ([link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.memory_usage.html)) to be ~1.5Gb. The benchmark timings translate to ~4.5Mb/s. The cloudfetch blog is stating 500 MB/s. 

I [ran a speedtest](https://www.speedtest.net/apps/cli) on my compute instance and confirmed bandwidth was not the problem (a comfortable 8000 MB/s down and 400 MB/s up). 

Together with the infra engineer we were not being able to detect anything wrong with the databricks / cloudfetch setup, so I tried something else.

## arrow-odbc

Given the first speedup in the databricks SQL connector was due to using arrow tables, I searched and found the [arrow-odbc-py](https://github.com/pacman82/arrow-odbc-py) project. It “Reads Apache Arrow batches from ODBC data sources in Python”. 

| Method | Speed |
| --- | --- |
| Baseline | 6m57s |
| .fetchall_arrow | 3m38s |
| Cloudfetch | 4m24s |
| arrow-odbc | 1m25s |

That is a very nice speedup! This connection allows you to tweak the batch sizes, so as a proper data scientist I decided to run some more benchmarks and optimize the batch size parameter. I highly recommend the [memo package](https://github.com/koaning/memo) for this kind of analysis. Tweaking the batch size helped but the performance gains were not huge across datasizes.

For reference, this is some sample code for connecting via `arrow-odbc`:

```python
import pandas as pd
from arrow_odbc import read_arrow_batches_from_odbc

def read_sql(query: str) -> pd.DataFrame:
	reader = read_arrow_batches_from_odbc(
	            query=f"select * from your_table",
	            connection_string=get_your_connection_string(),
	            batch_size=20_000,
	)
	if not reader:
			return None
	
	dfs = []
	for arrowbatch in reader:
	    # Process arrow batches
	    dfs.append(arrowbatch.to_pandas(timestamp_as_object=True))
	if dfs:
	    return pd.concat(dfs, ignore_index=True)
	else:
	    return None
```

## Turbodbc

Another project that should be mentioned is [Turbodbc](https://turbodbc.readthedocs.io/en/latest/). It’s a python project which uses many optimizations (like arrow and batched queries) to offer superior performance over ‘vanilla’ ODBC connections.

| Method | Speed |
| --- | --- |
| Baseline | 6m57s |
| .fetchall_arrow | 3m38s |
| Cloudfetch | 4m24s |
| arrow-odbc | 1m25s |
| turbodbc | 1m10s |

Using the [memo package](https://github.com/koaning/memo) I tracked many tweaks to the settings, including [using asyncio](https://turbodbc.readthedocs.io/en/latest/pages/advanced_usage.html#asynchronous-input-output), `strings_as_dictionary` and `adaptive_integers`. The gains were minor but still worth exploring the combinations.

A downside of Turbodbc however is that you need additional software to compile the C++ code that is [required for installation](https://turbodbc.readthedocs.io/en/latest/pages/getting_started.html#installation). The package is also [available on conda](https://anaconda.org/conda-forge/turbodbc) but installation was still less straightforward.

## databricks-sql-connector

As of July 2023, the `databricks-sql-connector` [v2.8.0 release](https://github.com/databricks/databricks-sql-python/releases/tag/v2.8.0) supports cloudfetch. The option is not well documented but here's a reference implementation:

```python
import os
from databricks import sql
import pandas as pd

def read_sql(query: str) -> pd.DataFrame:
   connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
        use_cloud_fetch=True, # <-- Make sure to specify this, as default is False
    )
    cursor = connection.cursor()
	try:
		cursor.execute(query)
		table = cursor.fetchall_arrow()
		df = table.to_pandas(timestamp_as_object=True)
	finally:
		cursor.close()
		connection.close()
	return df
```

This method is almost as fast as `turbodbc`:

| Method | Speed |
| --- | --- |
| Baseline | 6m57s |
| .fetchall_arrow | 3m38s |
| Cloudfetch | 4m24s |
| arrow-odbc | 1m25s |
| databricks-sql-connector | 1m19s |
| turbodbc | 1m10s |

In my benchmarks, the method is much faster for smaller queries as well. If I compare the time taken with the size of the final pandas dataframe, I measure speeds of ~80-150 MB/s (depending on the size of the query).

## The final setup

Ease and reliability of installation is important. We need the connection to databricks during various batch deployments, CI/CD builds and in different development environments. So we decided not to go for `turbodbc` and instead opt for the much simpler to setup (and only slightly slower) [databricks-sql-connector](https://github.com/databricks/databricks-sql-python).

## Conclusion

Investing some time in optimizing frequent and slow operations definitely pays off. In this case queries are >4x faster.

The rabbit hole is much deeper however and the potential for further speedups is still significant. For example, Databricks delta tables use parquet files under the hood, which means it might be possible to hook them up to [duckdb](https://duckdb.org/), which in turn has many optimizations for fetching data. And there’s the apache arrow flight project [announced in February 2022](https://arrow.apache.org/blog/2022/02/16/introducing-arrow-flight-sql/) that aims to get rid of many intermediate steps and natively support columnar, batched data transfers.