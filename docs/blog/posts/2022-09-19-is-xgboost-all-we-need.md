---
date: 2022-09-19 8:00:00
slug: is-xgboost-all-we-need
authors:
  - timvink
---

# Is XGBoost really all we need?

If you have experience building machine learning models on tabular data you will have experienced that gradient boosting based algorithms like [catboost](https://catboost.ai/), [lightgbm](https://lightgbm.readthedocs.io/) and [xgboost](https://xgboost.readthedocs.io/) are almost always superior.

It's not for nothing Bojan Tunguz (a quadruple kaggle grandmaster employed by Nvidia) states:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">XGBoost Is All You Need<br><br>Deep Neural Networks and Tabular Data: A Survey<a href="https://t.co/Z2KsHP3fvp">https://t.co/Z2KsHP3fvp</a> <a href="https://t.co/uh5NLS1fVP">pic.twitter.com/uh5NLS1fVP</a></p>&mdash; Bojan Tunguz (@tunguz) <a href="https://twitter.com/tunguz/status/1509197350576672769?ref_src=twsrc%5Etfw">March 30, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

... but aren't we all fooling ourselves?

<!-- more -->

## The illusion of progress

In his 2006 paper [Classifier Technology and the Illusion
of Progress](https://projecteuclid.org/journals/statistical-science/volume-21/issue-1/Classifier-Technology-and-the-Illusion-of-Progress/10.1214/088342306000000060.full) David J. Hand argues that the 'apparent superiority of more sophisticated methods may be something of an illusion' and that 'simple methods typically yield performance almost as good as more sophisticated methods' and the difference in performance 'may be swamped by other sources
of uncertainty that generally are not considered in the classical supervised
classification paradigm'.

Let's dive into his main arguments:

### 1. Law of marginal improvements

The extra performance achieved by more sophisticated algorithms beyond those from simple methods, is small. 

There is a law of diminishing returns: Simple models will learn the basic, most apparent data structures that lead to greater improvements in predictive performance, while newer approaches learn the more complicated structures.

The same goes for the number of features. For regression: If a set of features are all somewhat correlated with the target, there will be mutual correlation between them. We usually select those features with the highest target correlation first (for example with [MRMR](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)). Hand shows that (for regression) even with a low mutual correlation a small number of predictors will already explain most of the variance in the target. For classification a similar argument can be made: each feature added has a smaller maximum reduction in misclassification rate.

So the proportion of gains attributable to the early steps is big: both in data structures learned by simple models, and with the first couple most promising features.

### 2. Simple classifiers are very effective

"In many problems the bayes error rate is high, meaning that no decision surface can separate the distributions of such problems very well." But it is "common to find that the centroids of the predictor variable distributions of the classes are different, so that a simple linear surface can do surprisingly well as an estimate of the true decision surface.".

To prove this, Hand not only refers to other studies, but also does his own. He takes 10 different datasets and defines the achieve performance in predictive accuracy for each as the difference between the best known model (in 2006 of course) and a majority-class vote baseline model. He then shows that simple classifiers are able to reach 85-95% of the achievable improvement for those datasets, with most over 90%.

Another argument used for this is the "flat maximum effect" for linear models. When the correlations between features are high, often most of the gains can be made by simply assigning equal weights to each feature, without spending any time on optimizing the weights. In order words: quite large deviations from the optimal set of weights would yield performance not much worse than the optimal weights. This is because the simple average would be highly correlated with any other weighted sum: the choice of weight makes little difference to the scores.

### 3. Population bias

Hand calls it "design sample selection" and basically argues it's unlikely our assumption holds that our train splits are representative of the distribution we will use our model on. 

If there's a time component, the future data our model will use in production is unlikely to have the same distribution as our historic training data (f.e. predicting loan defaults of new customers using previous customers as training data). Things change in the future, and if you want your model to keep working, don't make it fit the training data too well. Hand quotes Eric Hoffer saying "In times of change, learners inherit the Earth, while the learned find them-selves beautifully equipped to deal with a world that no longer exists.". 

If there's a sampling component, it's unlikely the population is the same. For example medical screening of new cases using a sample of previous patients: it's hard to find a sample where you are confident it sufficiently represents the entire population. Another example is in credit risk, where you are using training data of _accepted_ client loans to predict whether new customer will default, but you will never know what historic _rejected_ clients would have done (a problem which even has it's own name, _reject inference_.)

So again, simple models pick up the bigger patterns which generalize better, while complex models are more susceptible to population bias and 'effort spent on overrefining the model is probably wasted effort'. Even computationally expensive sampling techniques such as repeated, grouped K-fold cross validation (see Sebastian Raschka's 2018 paper [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808)) cannot not fix population bias.


### 4. Problem bias

We often just assume there are no errors in the labels (`y`). Errors in the label means the variance of the estimated decision surface will be greater: it's better to stick to simpler models because the decision surface is flatter. 

The [2021 paper by Northcut et al](https://arxiv.org/abs/2103.14749) confirm Hand and actually show errors in tests sets are numerous and widespread: on average ~3.4% across _benchmark ML test sets_! (see [labelerrors.com](https://labelerrors.com/about)). They conclude:

> "Surprisingly, we find lower capacity models may be practically more useful than higher capacity models in real-world datasets with high proportions of erroneously labeled data." 

Label definition is another example of problem bias. The rules to create labels can be somewhat arbitrary, such as 3 months missed payments to define a 'default', grades above an 8 to define 'gifted' students and business rules to define 'good' and 'bad' customers. If the definitions are somewhat arbitrary, they can change over time, so there is no point in creating an overrefined model. 

When building models, often the metric used to select models is different than the metric to optimize the model (loss function), and both are different from the performance metric that actually matters in production. This simplification is often used (cost-based metrics are not common) but we should accept it introduces a bias. 

### 5. Flawed benchmarks

Finally, Hand argues that all algorithms require fine-tuning, which requires expertise in the methods. Obviously inventors know their own method best, so their papers and benchmarks are biased. Furthermore, average performance on standardized datasets like [UCI](https://archive.ics.uci.edu/ml/datasets.php) are not representative for real-world machine learning problems. Different algorithms also require different pre-processing to perform best. And finally, as for the reasons explained above, "small differences in error rate ... may vanish when problem uncertainties are taken into account".

Another argument from the paper is that method 1 may be 3x better than method 2, but if the error rate is already (very) low this might correspond to only a small proportion of the new data points. In order words, making 5 or 15 mistakes doesn't really matter when you're right 10.000 times. 

## Counter arguments

Hand offers many convincing arguments, but does that mean we should always go for simple models? Is XGBoost _not_ all we need?

Things have changed a lot since 2006. 

Let's start with the _law of marginal improvements_ (1). This is true of course, but anno 2022 our problems usually have a lot more data available. I would argue that for many real-life problems with large datasets GBM algorithms (like XGBoost) are often on the sweet spot between model complexity and the ability to generalize. Let's consider even more complex algorithms. The [2022 paper by Borisov et al.](https://arxiv.org/pdf/2110.01889.pdf) has an overview of the state-of-the-art deep learning methods for tabular data. They found that "In general, however, our results are consistent with the inferior performance of deep learning techniques in comparison to approaches based on decision tree ensembles (such as gradient boosting decision trees) on tabular data that was observed in various Kaggle competitions".

Then there's the _flawed benchmarks_ (5) argument. Creating different model and data preprocessing pipelines is now _much_ easier because of better ML tools. Consider that [scikit-learn](https://scikit-learn.org/stable/) was created in 2007 and had it's first beta release in 2010 ([wiki](https://en.wikipedia.org/wiki/Scikit-learn)). Regarding bias introduced by hyperparameter tuning: We now have much more compute power for hyperparameter tuning and better tools to run them: We can easily create fair benchmarks by using more compute. In the 2022 paper [Why do tree-based models still outperform deep
learning on tabular data?](https://arxiv.org/pdf/2207.08815.pdf) Grinszatjn et al build a standardized benchmark on many different datasets and give each algorithm a fair 20.000 hours of hyperparameter tuning. Not only is their code open source (Github only in [started in 2007](https://en.wikipedia.org/wiki/GitHub)!), there are many great tools to build comparable benchmarks with hyperparameter tuning (like [optuna](https://optuna.readthedocs.io/en/stable/) or the tool the authors used [weights&biases](https://wandb.ai/site)). Furthermore, [Kaggle](https://www.kaggle.com/) (only [started in 2010](https://en.wikipedia.org/wiki/Kaggle)) was built around data science competitions but contains a huge amount of reproducable, diverse experiments on many different datasets. I would argue that the arguments for flawed benchmarks do not hold any longer in 2022.

## Conclusions

Hand argued in 2006 that simple methods typically yield performance almost as good as more sophisticated methods. In 2022 we have larger datasets, more compute, better tooling and libraries that make it easier to write data and model pipelines. They help to address some of Hand's arguments and many benchmarks confirm that Gradient boosting based methods like XGBoost offer superior performance on tabular data.

There are still important lessons to be learned from Hand. His arguments on _population bias_ (3) and _problem bias_ (4) are as relevant as ever. And _simple classifiers are_ still _very effective_ (2) and have a place in our toolkit.

A data scientist should think and study the problem domain and the use of the data deeply before implementing a model. Some advice:

- If there are enough sources of uncertainty / variability, the 'principle of parsimony' (aka [occam's razor](https://www.geeksforgeeks.org/occams-razor/)) applies: stick to simple models
- If model explainability is a thing -- consider simple models
- If model deployment is difficult -- consider simple models
- Consider _population bias_ seriously. For example, on which data should you *not* make a prediction? See the talk [How to constrain artificial stupidity](https://www.youtube.com/watch?v=Z8MEFI7ZJlA).
- Don't trust your labels, _especially_ when using complex algorithms. Consider using tools like [cleanlab](https://github.com/cleanlab/cleanlab) ([intro video](https://calmcode.io/bad-labels/introduction.html)) and [doubtlab](https://github.com/koaning/doubtlab) to detect them.
- Model monitoring and model retraining pipelines are important, and even more so for complex algorithms
- Building robust models is all about error analysis
- 'Good enough' is good enough. Don't go crazy optimizing a model. Grid searching hyperparameters can be a trap. See the excellent blog series [Gridsearch is not enough](https://koaning.io/posts/enjoy-the-silence/) by Vincent Warmerdam.

So is XGBoost all you need? Probably not.
