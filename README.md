# ucl_irdm2017_project2_group1

This is the accompanying repository for Group 1's Learning to Rank project for UCL COMPGI15 Information Retrieval & Data Mining 2017 Group Project (Option 2).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development purposes.

### Prerequisites

Our model is written in Python.

The following modules are required to run our data exploration, models and evaluation:
* numpy
* pandas
* scipy
* random
* math
* pickle
* collections
* functools
* itertools
* matplotlib
* seaborn
* tensorflow 1.0.1
* scikit-learn
* itertools
* skbayes

### Installing

Clone the repository:
```
https://github.com/gpeake/ucl_irdm2017_project2_group1
```

For details on how install **sklearn-bayes**, please see: https://github.com/AmazaspShumik/sklearn-bayes

For details on how install **tensorflow**, please see: https://www.tensorflow.org/install/

For Python package installations in general, please see: https://packaging.python.org/installing/


## Usage

### Modules

* `data_load.py` contains a function to convert the MSLR-10K text files into Pandas dataframes and save as CSVs. This function only needs to be run once, after which the CSV files can be directly loaded.
* `utils` contains functions which are shared between models, namely rank_query required for evaluation metrics
* `evals.py` contains functions to evaluate our models: DCG@N, NDCG@N, ERR, Mean NDCG, Mean ERR, Accuracy

### Notebooks

We have included a number of 'plug and play' notebooks, demonstrating how our models can be used to predict rankings for the MSLR-10K dataset and how to evaluate the results.

* `data_exploration` contains statistics of the dataset
* `logistic_regression_classifier` contains training, cross-validation, evaluation, regularisation and feature-engineering details for our logistic regression classifier
* `deep_nn`

### Models

The models folder contains pickle files of pre-trained models to avoid re-running. The code to load and evaluate these models can be found in the notebooks.

### Data

The MSLR-10K data can be downloaded from the [Microsoft website](https://www.microsoft.com/en-us/research/project/mslr/).

The data is in .txt file format. In our notebooks, we provide code to convert these files into dataframes (and save as CSVs). Once the CSVs are generated, the files should be stored at the top-level of the repository, i.e. in the ucl_irdm2017_project2_group1 parent folder, to work with our 'plug and play' notebooks.

## Authors

* **Benjamin Ajayi-Obe**
* **Georgina Peake**
* **Amy Roberts**
* **Ruadhan Stokes**

## Acknowledgments

* [Microsoft MSLR-10K dataset](https://www.microsoft.com/en-us/research/project/mslr/)
* The Lemur project - [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/)
* [Python package for Bayesian Machine Learning with scikit-learn API](https://github.com/AmazaspShumik/sklearn-bayes)
* Our logistic regression classifier is based on: Li, P., Burges, C. and Wu, Q. 2007. McRank: Learning to rank using multiple classification and gradient boosting. *NIPS'07 Proceedings of the 20th International Conference on Neural Information Processing Systems*. 2007: 897-904
* Our deep neural network is based on: Covington, P., Adams, J. and Sargin, E. 2016. Deep Neural Networks for YouTube Recommendations. *RecSys '16 Proceedings of the 10th ACM Conference on Recommender Systems*. 2016: 191-198
* UCL COMPGI15 Professors and Teaching Assistants
