---
date: 2019-01-11
slug: google-cloud-functions
authors:
  - timvink
---

# How to develop Python Google Cloud Functions

I've been using Google's [Cloud Functions](https://cloud.google.com/functions/) for a project recently. My use case was building a small webscraper that periodically downloads excel files from a list of websites, parses them and write the structured data into a BigQuery database. In this blogpost I'll show you why going serverless with cloud functions is so amazing and cost-effective. And I'll discuss some practical problems that have solutions
you can't easily find in the [documentation](https://cloud.google.com/functions/docs/) or stackoverflow. 

<!-- more -->

In this blog:

- [The basics](#the-basics)
- [Integrate with Cloud Storage](#integrate-with-cloud-storage)
- [Debugging your Cloud Functions](#debugging-your-cloud-functions)
- [Testing Cloud Functions locally](#testing-cloud-functions-locally)
- [Troubleshooting](#troubleshooting)
- [Bonus: Static sites with serverless backend](#bonus-static-sites-with-serverless-backend)
- [Conclusion](#conclusion)

## The basics

> Cloud Functions are small pieces of code that execute in an event-driven manner. They do one small thing very efficiently in reaction to a trigger — usually an HTTP request. The neat thing is you manage zero infrastructure and usually only pay for the execution of your function and a few seconds of compute. You may have heard of other Functions-as-a-Service offerings such as AWS Lambda. ([source](https://medium.com/@timhberry/getting-started-with-python-for-google-cloud-functions-646a8cddbb33))

This means you can write a python 3.7 function and deploy it without handling infrastructure, OS or networking. Google takes care of scaling, whether you call your function once a month or millions of times a day. You only pay per 100ms of functions running, which means using Cloud Functions is much cheaper than deploying on your own servers (see [pricing](https://cloud.google.com/functions/pricing)), with a generous free tier as well.

Let's try it out. Create a project on [Cloud Console](http://console.cloud.google.com) and setup billing (don't worry; all examples below are in free tier). Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/) (`brew cask install google-cloud-sdk` on macOS), login and setup some defaults:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/region europe-west1
gcloud config set compute/zone europe-west1-b
```

Cloud Functions have their own independent environment, so we put them in
their own folder (tip: use the function name). The folder should have a `main.py` file
and optionally if you need extra packages a `requirements.txt` file. Here is a simple project structure:

```bash
tree myproject
#> myproject
#> ├── README.md
#> └── functions
#>     └── hello_world
#>         └── main.py
```

Cloud functions are called either by events in your project cloud environment, or
by certain triggers ([docs](https://cloud.google.com/functions/docs/concepts/events-triggers)). We'll focus on HTTP trigger functions. Google uses [Flask](http://flask.pocoo.org/) in their python runtime to handle incoming requests. This means the input for our function should be Flask request object, and the output should be a valid Flask response:

```python
# main.py
def hello_world(request):
  return 'Hello World!'
```

Deployment is a breeze:

```bash
cd functions/hello_world
gcloud beta functions deploy hello_world \
  --runtime python37 \
  --trigger-http \
  --region europe-west1
#> Deploying function (may take a while - up to 2 minutes)...done.
#> availableMemoryMb: 256
#> entryPoint: hello_world
#> httpsTrigger:
#>   url: https://us-central1-<YOUR PROJECT_ID>.cloudfunctions.net/hello_world
#> ...
```

Visit the httpsTrigger URL to view the output in your browser, or use `gcloud functions call hello_world` in your terminal.

## Integrate with Cloud Storage

As an example we'll demonstrate how to integrate with a [Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) and create a cloud function that downloads a file for us. I'll use `gsutil` to create
a bucket and upload an image of the SpaceX starship test vehicle.  

```bash
gsutil mb -l europe-west1 gs://<your_bucket_name>
gsutil cp ~/Downloads/starship.jpeg gs://<your_bucket_name>
#> / [1 files][  5.9 KiB/  5.9 KiB]
#> Operation completed over 1 objects/5.9 KiB.
```

We could just make this bucket public, or share a _signed URL_ to download this specific file. But for practice, we'll write the cloud function that will let us download the file. To get the image from the bucket into the python cloud function environment, we could
use `tempfile.gettempdir()` to download it to the `/tmp` directory, an _in-memory_ mount of the cloud function ([source](https://stackoverflow.com/a/42719827)). Instead, we'll use `io.BytesIO` to create an object in memory directly. We'll use the `flask.send_file()` to return the image as a file:

```python
# main.py
from io import BytesIO
from flask import Flask, request, send_file
from google.cloud import storage
storage_client = storage.Client()

def download_file(request):
    bucket = storage_client.get_bucket('<your bucket name>')
    blob = bucket.get_blob('starship.jpeg')
    file = BytesIO(blob.download_as_string())
    return send_file(file,
        attachment_filename = blob.name,
        as_attachment=True,
        mimetype='image/jpeg')
```

To interface with the bucket from python, I'm using the [google-cloud-storage](https://pypi.org/project/google-cloud-storage/) package. This is not installed in the python runtime so we need to add it to a `requirements.txt` file:

```yaml
# requirements.txt
google-cloud-storage
```

You file structure should now look like:

```bash
tree myproject
#> myproject
#> ├── README.md
#> └── functions
#>     ├── download_file
#>     │   └── main.py
#>     │   └── requirements.txt
#>     └── hello_world
#>         └── main.py
```

And then deploy with:

```bash
cd functions/download_file
gcloud beta functions deploy download_file \
  --runtime python37 \
  --trigger-http \
  --region europe-west1
```

Visit the URL to download the image!

## Debugging your Cloud Functions

So far it's easy sailing. But what happens if you cloud functions starts returning
`Error: could not handle the request` ?

To find the mistake, you have some options. In your project's cloud console, go to
cloud functions and click on your function.

- In the _general_ tab you can see the latest errors.
- In the _testing_ tab you can run your functions and see some logs.
- The _View Logs_ button shows python logs of your function.

This workflow becomes annoying very quickly: it can take up to 2 minutes to deploy
a new cloud function. And it does not always pinpoint the problem. In my logs I had a
`finished with status: 'connection error'` and a test run returned that `Error: cannot communicate with function.`, both of which did not help me find the error. But there's a better way!

## Testing Cloud Functions locally

Google [describes](https://cloud.google.com/functions/docs/bestpractices/testing) how to use `unittest` to mock Flask and test a HTTP-trigger python function. This is great for unit testing, but for development I preferred to write a simple Flask app so I could call my functions locally:

```python
# DevelopmentFlaskApp.py (in root of project)
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceaccount.json"

from flask import Flask, request, render_template, send_from_directory
from functions.download_file import main as f_download_file

app = Flask(__name__)

@app.route('/download_file')
def flask_download_file():
    return f_download_file.download_file(request)

if __name__ == '__main__':
    app.run()
```

Notice I set an environment variable pointing to a JSON. Our `google-cloud-storage` package uses these credentials to authenticate. We want to use the same permissions as the cloud function would have at runtime:

> At runtime, Cloud Functions uses the service account PROJECT_ID@appspot.gserviceaccount.com, which has the Editor role on the project. You can change the roles of this service account to limit or extend the permissions for your running functions. ([source](https://cloud.google.com/functions/docs/concepts/iam))

Let's download these credentials to the root of our project (and if you use git, don't forgot to update `.gitignore`):

```bash
gcloud iam service-accounts keys create serviceaccount.json \
  --iam-account <PROJECT_ID>@appspot.gserviceaccount.com
```

To run our flask app:

```bash
pip install flask
export FLASK_APP=DevelopmentFlaskApp.py
export FLASK_ENV=development
flask run
```

You can then visit `localhost:5000/download_file` to see if you cloud function works, without having to wait 2 minutes to deploy! And with the credentials json already downloaded, you could also opt to develop some functionality in a notebook.

## Troubleshooting

In practice, you can run into all sorts of problems. Some of them were hard to debug and fix, so sharing them here:

### Zombie deployments

I had a lot of trouble getting one of my cloud functions to work. It worked perfect *locally*.
I spent a lot of time reading about permissions, but it turns out _the function was not overwritten after deployment_ (!!). Luckily, I'm not alone in this problem ([github issue](https://github.com/firebase/functions-samples/issues/118)). Even deleting my function wouldn't stop the URL from working. The solution for me was re-deploying using a different name.

### Setting permissions

If you happen to have a permission problem, it's fairly easy to solve. In Google Cloud, a _policy binding_ is where a _member_ (user or serviceaccount) gets attached to a _role_ (which contains 1 or more _permissions_). Remember, for cloud functions the _member_ will be the app engine service account. Next, find an appropriate [predefined role for a cloud product](https://cloud.google.com/iam/docs/overview#iam_support_for_cloud_platform_services). Here's an example of adding a _policy binding_ using `gcloud`:

```bash
gcloud projects add-iam-policy-binding \
        --member serviceAccount:<PROJECT_ID>@appspot.gserviceaccount.com \
        --role roles/storage.admin
```

Sometimes, `gcloud` will ask you to enable a certain API. Here's an example for BigQuery:

```bash
gcloud services enable bigquery-json.googleapis.com
```

### Connection reset by peer

You might find an `ConnectionResetError: [Errno 104] Connection reset by peer` error in your logs, and it's not helpful at all. In my case, it had to do with creating clients for storage buckets and bigquery.
This [SO post](https://stackoverflow.com/questions/52129628/python-google-cloud-function-connection-reset-by-peer) confirmed my suspicion, that in ~10% of the cases creating the connection throws a connection reset error. The solution is a simple retry with some random wait time:

```python
from google.cloud import storage
from google.cloud import bigquery
from retrying import retry

@retry(stop_max_attempt_number=3, wait_random_min=1000, wait_random_max=2000)
def get_google_client(type):
    if type == 'storage':
        return storage.Client()
    if type == 'bigquery':
        return bigquery.Client()

storage_client = get_google_client('storage')
```

### Finished with status 'timeout'

If you call your HTTP function and get the very generic `Error: could not handle the request`, dive into the logs. You might find a `Function execution took 60003 ms, finished with status: 'timeout'`. I had missed it from reading the documentation, but cloud functions are capped to at most 60 seconds execution time. You can [increase the timeout](https://cloud.google.com/functions/docs/concepts/exec#timeout) to up to 9 minutes. Alternatively, you need to split up your function. In my case, I had to create a separate cloud function to download and process each file on a webpage.

### Error with status 500

Another error that took me some time to figure out. I was invocating many functions at the same time using `asyncio`, and got really vague status 500 errors. Cause: Google Cloud Functions have many different [types of limits and quotas](https://cloud.google.com/functions/quotas), and: *A function returns an HTTP 500 error code when one of the resources is over quota and the function cannot execute*. For me it was that I was not using global variables to reuse objects in future invocations (see the [best practices](https://cloud.google.com/functions/docs/bestpractices/tips)). Another way to solve it could be to move to background functions listening to pub/sub events, or increasing the quotas for your project.

## Bonus: Static sites with serverless backend

In my case I was [hosting a static website with app engine](https://cloud.google.com/appengine/docs/standard/python/getting-started/hosting-a-static-website) and using cloud functions on the backend. I wanted to test and develop the site locally as a Flask app. In order to change the URL locally, you can use some javascript:

```javascript
<script>
jQuery(document).ready(function(){
  var gcf_download_file = "https://us-central1-PROJECT_ID.cloudfunctions.net/download_file"
  if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
    var gcf_download_file = "/download_file"
  }
  $('#my-download-link').attr("href", gcf_download_file);
});
</script>
```

## Conclusion

Cloud functions are extremely flexible and offer myriad possibilities. And because each invocation of a function has it's own environment, you can easily scale your code horizontally. If you want to build something a bit more complex, I recommend looking using [pub/sub](https://cloud.google.com/functions/docs/calling/pubsub) and letting cloud functions process a message queue.

Good luck!
