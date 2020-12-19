******** Please read the readme file before running the program ********

Group Project submitted by Lukas Tillmann for the Course "Programming With Advanced Computer Languages" at the
University of St. Gallen (Fall Semester 2020). Written 100% in Python.

Description:
Based on clinical data, the program distinguishes between H1N1 (Influenza) and SARS-CoV-2 (COVID-19) cases, applying
three different supervised machine learning algorithms. This is to emphasize not only different methods of machine
learning, but also the general trade-off between accuracy and (time) efficiency of these algorithms. One can define the
split ratio of the dataset for training and testing of the models. For every model, predicted class labels, true class
labels, evaluation statistics, a classification report, a confusion matrix and the corresponding
Receiver-Operator-Characteristic (ROC) curve are given, making the models comparable. Lastly, the program saves all
plot results as .png files in an output folder.

Please make sure you have the following modules installed before running the files:
pandas, scikit-learn, numpy, matplotlib, seaborn, missingno, palettable

It is divided into the following parts:
- Data Import, Preparation & missing data Imputation
- Data Visualization
- K-Nearest-Neighbours (kNN) Classifier (automatically calculates best no. of k)
- Logistic Regression
- Multi-layer Perceptron Classifier (automatically searches for best parameters)

Run Instructions:
- make sure python is installed on your computer (Disclaimer: run with Python 3.9 as base interpreter)
- install needed modules (you can use pip or set up your own interpreter virtual environment)
- to run from finder (explorer), select .py file you want to run and open with Python Launcher
- to run in terminal, find path where project is saved, open terminal and type: python /yourpath/fileyouwanttorun.py

Limitations & research outlook:
- a lot of the data needed to be imputed as dataset is incomplete, therefore model not necessarily applicable in reality
- models are trained to only differentiate between Influenza and COVID-19, therefore prediction will always be one of
  the two, even even patient is healthy or has a different disease
- an input screen to predict new cases using the trained models is not yet included

Sources:
Clinical Data retrieved from https://github.com/yoshihiko1218/COVID19ML/blob/master/UsedCombined.txt