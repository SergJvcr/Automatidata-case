import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier, plot_tree # the last one plots feature importance 


# The purpose of this model is to find ways to generate more revenue for taxi cab drivers.
# The goal of this model is to predict whether or not a customer is a generous (щедрый) tipper.


# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)

# Load dataset into dataframe
df0 = pd.read_csv('google_data_analitics\\2017_Yellow_Taxi_Trip_Data.csv')
# Import predicted fares and mean distance and duration from previous step of work
nyc_preds_means = pd.read_csv('google_data_analitics\\nyc_preds_means.csv')

print(df0.head(10))
print(nyc_preds_means.head(10))

# Merge datasets into one
df0 = df0.merge(right=nyc_preds_means, left_index=True, right_index=True)

# Feature engineering
print(df0.info())
print(df0.describe(include='all'))

# Subset the data to isolate only customers who paid by credit card
# (We know from previous step that customers who pay cash generally have a tip amount of $0)
mask_credit_card = df0['payment_type'] == 1
df1 = df0[mask_credit_card]

# Create tip % col
df1['tip_percent'] = round(df1['tip_amount'] / (df1['total_amount'] - df1['tip_amount']), 3)
# Create 'generous' column - the target variable
df1['generous'] = df1['tip_percent']
df1['generous'] = (df1['generous'] >= 0.2)
df1['generous'] = df1['generous'].astype(int)

# Create day column
# Convert pickup and dropoff cols to datetime
df1['tpep_pickup_datetime'] = pd.to_datetime(df1['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df1['tpep_dropoff_datetime'] = pd.to_datetime(df1['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

print(df1.dtypes)

# Create a 'day' column
df1['day'] = df1['tpep_pickup_datetime'].dt.day_name().str.lower()

# Create time of day columns
# (Each column should contain binary values (0=no, 1=yes) that indicate 
# whether a trip began (picked up) during the following times)
# Create 'am_rush' col
df1['am_rush'] = df1['tpep_pickup_datetime'].dt.hour
# Define 'am_rush()' conversion function [06:00–10:00)
def am_rush_func(hour):
    if 6 <= hour['am_rush'] < 10:
        return 1
    else:
        return 0
df1['am_rush'] = df1.apply(am_rush_func, axis=1)

# Create 'daytime' col
df1['daytime'] = df1['tpep_pickup_datetime'].dt.hour
# Define 'daytime()' conversion function [10:00–16:00)
def daytime_func(hour):
    if 10 <= hour['daytime'] < 16:
        return 1
    else:
        return 0
df1['daytime'] = df1.apply(daytime_func, axis=1)

# Create 'pm_rush' col
df1['pm_rush'] = df1['tpep_pickup_datetime'].dt.hour
# Define 'pm_rush()' conversion function [16:00–20:00)
def pm_rush_func(hour):
    if 16 <= hour['pm_rush'] < 20:
        return 1
    else:
        return 0
df1['pm_rush'] = df1.apply(pm_rush_func, axis=1)

# Create 'nighttime' col
df1['nighttime'] = df1['tpep_pickup_datetime'].dt.hour
def nighttime_func(hour):
    if 20 <= hour['nighttime'] < 24 or 0 <= hour['nighttime'] < 6:
        return 1
    else:
        return 0
df1['nighttime'] = df1.apply(nighttime_func, axis=1)

# Create 'month' column
df1['month'] = df1['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
print(df1.head(5))

# Drop redundant and irrelevant columns
drop_columns = ['ID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
             'payment_type', 'trip_distance', 'store_and_fwd_flag', 'payment_type',
             'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
             'improvement_surcharge', 'total_amount', 'tip_percent']
df1 = df1.drop(drop_columns, axis=1)
print(df1.info())

# Variable encoding
# Many of the columns are categorical and will need to be dummied (converted to binary). 
# Some of these columns are numeric, but they actually encode categorical information, 
# such as RatecodeID and the pickup and dropoff locations

# Define list of cols to convert to string
columns_to_str = ['RatecodeID', 'PULocationID', 'DOLocationID']
# Convert each column to string
for column in columns_to_str:
    df1[column] = df1[column].astype(str)

print(df1.info())

# Convert categoricals to binary
df_prepared = pd.get_dummies(df1, drop_first=True)
df_prepared.info()
print(f'The prepared dataset has {df_prepared.shape[0]} columns and {df_prepared.shape[1]} rows.')

# Evaluation metric
# Examine the class balance of the target variable
print('The class balance of the target variable:', df_prepared['generous'].value_counts(normalize=True))

# Modeling
# Split the data
# Isolate target variable (y)
y = df_prepared['generous']
# Isolate the features (X)
X = df_prepared.drop(['generous'], axis=1)
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Random forest
# Instantiate the random forest classifier
random_forest_classifier = RandomForestClassifier(random_state=42)
# Create a dictionary of hyperparameters to tune 
# (We must use more possible values for each parameter in GridSearchCV 
# but it will be much longer and for saving time in this example
# we use only one value for every parameter)
cross_valid_params = {'max_depth':[None], 
                      'max_features':[1.0], 
                      'max_samples':[0.7], 
                      'min_samples_leaf':[1], 
                      'min_samples_split':[2], 
                      'n_estimators':[300]}
# Define a set of scoring metrics to capture
scoring_metrics = ['precision', 'recall', 'f1', 'accuracy']
# Instantiate the GridSearchCV object
rand_forest_search_params = GridSearchCV(estimator=random_forest_classifier, 
                                         param_grid=cross_valid_params, 
                                         scoring=scoring_metrics, 
                                         cv=5, 
                                         refit='f1')
rand_forest_search_params.fit(X_train, y_train)

# Examine the best score and the best parameters
print(f'The best F1 score for trained random forest model is {rand_forest_search_params.best_score_}')
print('The best parameters for our model are')
print(rand_forest_search_params.best_params_)

def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
    model_name (string): what you want the model to be called in the output table
    model_object: a fit GridSearchCV object
    metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                 'recall': 'mean_test_recall',
                 'f1': 'mean_test_f1',
                 'accuracy': 'mean_test_accuracy',
                 }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                        'precision': [precision],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy],
                        },
                       )

    return table

results = make_results('RF CV', rand_forest_search_params, 'f1')
print(results)

# Get scores on test data
random_forest_predictions = rand_forest_search_params.best_estimator_.predict(X_test)

def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
    model_name (string): Your choice: how the model will be named in the output table
    preds: numpy array of test predictions
    y_test_data: numpy array of y_test data

    Out:
    table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                        'precision': [precision],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy]
                        })

    return table

 # Get scores on test data
random_forest_test_scores = get_test_scores('RF test', random_forest_predictions, y_test)
results = pd.concat([results, random_forest_test_scores], axis=0)
print(results)

# XGBoost - next model, for comparing with random forest
# to choose the best of them model
# Instantiate the XGBoost classifier
xgboost_classifier = XGBClassifier(objective='binary:logistic', random_state=42)
# Create a dictionary of hyperparameters to tune
cross_valid_params = {'learning_rate': [0.1],
                      'max_depth': [8],
                      'min_child_weight': [2],
                      'n_estimators': [500]}
# Define a set of scoring metrics to capture
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
# Instantiate the GridSearchCV object
xgboost_search_params = GridSearchCV(estimator=xgboost_classifier, 
                                     param_grid=cross_valid_params, 
                                     scoring=scoring_metrics, 
                                     cv=5, 
                                     refit='f1')
xgboost_search_params.fit(X_train, y_train)
# Examine the best score and the best parameters
print(f'The best F1 score for trained random forest model is {xgboost_search_params.best_score_}')
print('The best parameters for our model are')
print(xgboost_search_params.best_params_)

# Call 'make_results()' on the GridSearch object
xgboost_cv_results = make_results('XGB CV', xgboost_search_params, 'f1')
results = pd.concat([results, xgboost_cv_results], axis=0)
print(results)

# Get scores on test data
xgboost_predictions = xgboost_search_params.best_estimator_.predict(X_test)

# XGB test results
xgboost_test_scores = get_test_scores('XGB test', xgboost_predictions, y_test)
results = pd.concat([results, xgboost_test_scores], axis=0)
print(results)

# Plot a confusion matrix of the model's predictions on the test data
# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, random_forest_predictions, labels=rand_forest_search_params.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rand_forest_search_params.classes_, 
                             )
disp.plot(values_format='')
plt.show()

# Feature importance - to inspect the features of your final model
importances = rand_forest_search_params.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test.columns)
rf_importances = rf_importances.sort_values(ascending=False)[:15]

fig, ax = plt.subplots(figsize=(8,5))
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()
plt.show()


