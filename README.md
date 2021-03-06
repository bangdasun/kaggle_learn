## kaggle_learn

This module is a collection for useful code in kaggle (or more data science general), includes general utility functions, feature engineering functions, model training / cross-validation / model stacking helper functions.


### Modules

#### Preprocessing

Numerical / categorical (tabular data) and text data preprocessing.

#### Feature Engineering

Some commonly used feature engineering methods used in kaggle competitions:

- group by on one / multiple categorical features and get the summary statistics of numerical features
- text meta features
- text similarity features
- time meta features
- ...

#### Models

Models not implemented in `sklearn`:

- NB-LR (Logistic Regression with Naive Bayes features)

Neural network templates for different tasks:

- text classification (RNN / CNN based)

#### Model Runner

Functions / classes that takes data and run specified models (with cross validation).

#### Feature Importance

- Permutation feature importance

#### Metrics

- customized `keras` callbacks

#### Utilities

Utility functions:

- reduce `pandas` dataframe memory
- timer / logger
- ...

### Usage

Clone this repository:

```
git clone https://github.com/bangdasun/kaggle_learn.git
```

Create environment to install required packages:

```
cd kaggle_learn
conda env create -f standard_env.yaml
```



