# ucl_irdm2017_project2_group1

This is the accompanying repository for Group 1's Learning to Rank project for UCL COMPGI15 Information Retrieval & Data Mining 2017 Group Project (Option 2).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* python
* numpy
* pandas
* matplotlib
* tensorflow 1.0.1
* scikit-learn

### Installing

Clone the repository:
```
https://github.com/gpeake/ucl_irdm2017_project2_group1
```

## Usage

### Modules

* `data_load.py` contains a function to convert the MSLR-10K text files into Pandas dataframes and save as CSVs. This function only needs to be run once, after which the CSV files can be directly loaded.
* `logistic_classifier` contains functions required to train our logistic regression classifier.
* `evals.py` contains functions to evaluate our models: DCG@N, NDCG@N, ERR

### Notebooks

We have included a number of 'plug and play' notebooks, demonstrating how our models can be used to predict rankings for the MSLR-10K dataset and how to evaluate the results.

* `data_exploration`
* `logistic_regression_classifier` 
* `deep_neural_network`

## Authors

* **Benjamin Ajayi-Obe**
* **Georgina Peake**
* **Amy Roberts**
* **Ruadhan Stokes**


## License



## Acknowledgments

* [Microsoft MSLR-10K dataset](https://www.microsoft.com/en-us/research/project/mslr/)
* The Lemur project
* Our logistic regression classifier is based on:
* Our deep neural network is based on: the 'Deep Neural Networks for YouTube Recommendations' paper by P. Covington, J. Adams, E. Sargin
