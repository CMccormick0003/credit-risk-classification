# Credit Risk Analysis Report

### An overview of the analysis: 
The purpose of this analysis is to rain and evaluate a model based on loan risk. 
This analysis uses a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers as a Healthy loan or a High-risk loan.

**** Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

The steps performed were:
- Split the Data into Training and Testing Sets
- Create a Logistic Regression Model with the Original Data
- Predict a Logistic Regression Model with Resampled Training Data
- 
THe tools used were:
- Pandas
- Python (pathlib module, imblearn.over_sampline module), 
- Scikit-learn (metrics module, model_selection module and linear_model)

## Background
The source data for this exercise are from the csv file: lending_data.csv.
Starter code was provided by the bootcamp in the notebook: credit_risk_classification.ipynb.
Definitions of healthy and high-risk loans:
- Healthy: A value of 0 in the “loan_status” column in the csv means the loan is healthy. 
- High-risk: A value of 1 in the "loan_status" column in the csv means the loan has a "high risk" of defaulting.

## Steps
### Split the Data into Training and Testing Sets
- Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
- Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
- Split the data into training and testing datasets by using train_test_split.

### Create a Logistic Regression Model with the Original Data
- Fit a logistic regression model by using the training data (X_train and y_train).
- Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.
- Evaluate the model’s performance by a) calculating the accuracy score of the model, b) generating a confusion matrix, and printing the classification report.
### How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?
Precision for healthy loans and high-risk was 1.0 and 0.85, respectively.   The model predicted healthy loans with much more accuracy than high-risk loans.  The overall weighted accuracy was 0.99.  I woudl have confidence using this model to predict both types of loans.


### Create a Logistic Regression Model with the Original Data
- Fit a logistic regression model by using the training data (X_train and y_train).
- Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.
- Evaluate the model’s performance by a) calculating the accuracy score of the model, b) generating a confusion matrix, and printing the classification report.

### Results
### Machine Learning Model 1: Logistic Regression based on Original Data
  - accuracy score: 99%
  - precision score: Healthy loan - 100%, High-risk loan - 85%
  - recall score of the machine learning model:Healthy loan - 99%, 
  - High-risk loan - 91%, Average for the model - 99%

### Machine Learning Model 1: Logistic Regression based on Resampled Training Data
This model used oversampled data.
  - accuracy score: 99%
  - precision score: Healthy loan - 100%, High-risk loan - 84%
  - recall score of the machine learning model:Healthy loan - 99%, 
  - High-risk loan - 91%, Average for the model - 99%

### Summary
Model 2 with oversampled data has the same precision to predict Healthy loans (100%) and slightly less precisions (84% compared to 85%) to predict High-risk loans compared to Model 1. Both models have high accuracy of 99%.  

Although both models have the same accuracy and similar precision, Model 2 used the resampled training data has better recall.  Recall represents the proportion of true positive predictions (correctly predicted positive instances) out of all actual positive instances. Recall is True Positives / (True Positives + False Negatives).  Model 2 also had a higher F1 score for High-risk loans compared to Model 1 (91% compared to 88%).  F1 score is a performance metric that combines both precision and recall into a single value.

Model 2 has a higher ability to correctly identify instances of High-risk loans out of all the actual instances of High-risk loans in the dataset.
In Model 2, 99% of the actual instances High-risk loans are correctly classified as High-risk by the logistic regression model. 
Model 2 has a very low rate of falsely classifying instances of High-risk loans as being Healthy.

MIsclassifying high-risk loans as healthy could have financial consequences.  Therefore, the model that I would recommend to management from this exercise is Model 2.
