# Prediction-of-Sepsis-from-Clinical-Data
Data Challenge of Umich STATS 503: Statistical Learning II: Modern Multivariate Analysis

# Dataset

The data consist of records from 21634 patients and has been split into a training set (with 15144 patients) and a test set (with 6490 patients).  Outcomes are provided for the training set, and are withheld for the test set.

## Prediction Variables

Approximately 40 variables were recorded at least once after the patient's admission to the ICU.  Several of these variables are general descriptors (such as age, gender, and the ICU unit), and the remainder are clinical vital signs and laboratory measurements, for which multiple observations may be available.  Each vital sign or laboratory measurement has an associated time-stamp indicating the elapsed time (in hours) of the measurement since ICU admission. 

Data was adapted from the following website:

https://physionet.org/content/challenge-2019/1.0.0/

Note: The data used here is a little different from website, without the outcome with time stamps.

# Quick start
```shell
# for running all the procedure
python resnet.py 

# for the individual task, run corresponding py scripts:
python data_clean_merge.py
python model_train.py
python model_apply.py
```

# Log:
The best params of XGBoost Classifier are {'colsample_bytree': 0.5, 'learning_rate': 0.06, 'max_depth': 9}

balanced accuracy is : 0.731

model score is : 0.907

precision score is: 0.737

AUC is: 0.885
