from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from data_preparation import X, y
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# ignore warnings
warnings.filterwarnings(action='ignore')

# show info
print('\nMultilayer-Perceptron Classifier')

# show notification
print('\nNOTE: because of its structure, this classifier may take a little time to compute all data!')

# split data in training and test sets
# set up infinite loop
while True:
    try:
        # ask for user input and save in variable
        eval_fraction = float(input('\nPlease enter the split ratio (float between 0 & 1) you want to use'
                                    ' (e.g. 0.3 for 30% testing, 70% training of classifier): ').strip())
        # if input is float between 0 and 1
        if 0 <= eval_fraction <= 1:
            # split dataset in training and evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=eval_fraction, random_state=1)
            # break infinite loop
            break
        # if input is float outside range of 0 and 1
        else:
            print('Error - Number not between 0 and 1')
    # catch value error when input is e.g. a string
    except ValueError:
        print('Error - Please enter a number!')

## MLPClassifier
# init classifier, maximum iterations 2000
mlp = MLPClassifier(max_iter=2000, random_state=1)
# define parameter-space to search
parameter_space = {
    'hidden_layer_sizes': [(5, 5, 5), (10, 10, 10), (5, 5, 5, 5)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'alpha': [1e-3, 0.05],
    'learning_rate': ['constant', 'adaptive']}
# search for optimal parameter settings in defined space with GridsearchCV, use all CPUs
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1)
# search for given data
clf.fit(X_train, y_train)
# best parameters found
best_parameters = clf.best_params_
print('\nBest parameters found:\n{}'.format(best_parameters))
# show network structure
print('\nNetwork Architecture Structure (no. of neurons per layer):\n{}\n'.format(best_parameters
                                                                                  ['hidden_layer_sizes']))
# use best parameters to predict classes of test data
multilayer_prediction = clf.predict(X_test)

## Evaluation
# show model statistics
print('\nPredicted class labels (0 = H1N1, 1 = SARS-CoV-2):\n{}'.format(multilayer_prediction))
print('\nTrue class labels (0 = H1N1, 1 = SARS-CoV-2):\n{}'.format(y_test.values))
print('\nPrediction Accuracy: {}\n'.format(metrics.accuracy_score(y_test, multilayer_prediction)))
print('Number of mislabeled points out of a total {} points: {}\n\n'.format(X_test.shape[0],
                                                                            np.sum(y_test != multilayer_prediction)))
# print classification report
print('Classification Report Prediction:\n\n{}'.format(classification_report(y_test, multilayer_prediction,
                                                                             target_names=['H1N1', 'SARS-CoV-2'])))
## Confusion Matrix
# initiation of confusion matrix
matrix = confusion_matrix(y_test, multilayer_prediction)
# init the plot
plt.figure(figsize=(7, 7))
# set axis labels
axis_labels = ['H1N1', 'SARS-CoV-2']
# plot confusion matrix heatmap
sns.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False, cmap='RdBu_r',
            xticklabels=axis_labels, yticklabels=axis_labels)
# add plot axis labels
plt.xlabel('[True label]')
plt.ylabel('[Predicted label]')
# add plot title
plt.title('Multilayer Confusion Matrix, layer structure = {}'.format(best_parameters['hidden_layer_sizes']))
# save as png file, remove all of whitespace around figure
plt.savefig('output/mlpclassifier_confusion matrix.png', bbox_inches='tight')
# show matrix
plt.show();

## ROC
# calculate false-positive/true-positive -rate
fpr, tpr, threshold = metrics.roc_curve(y_test, multilayer_prediction)
# calculate roc auc
roc_auc = metrics.auc(fpr, tpr)
# add title
plt.title('Receiver Operating Characteristic (Multilayer Perceptron Classifier)')
# plot roc curve and label
plt.plot(fpr, tpr, label='ROC curve')
# plot straight line to visualize random guessing and label
plt.plot([0, 1], [0, 1], 'r--', label='Random Chances')
# plot another label with only AUC value
plt.plot([], [], ' ', label='AUC (Area Under Curve) = {}'.format(roc_auc))
# plot legend at lower right location of figure
plt.legend(loc='lower right')
# define limits of x and y axis
plt.xlim([0, 1])
plt.ylim([0, 1])
# add x and y labels
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# save as png file, remove all of whitespace around figure
plt.savefig('output/mlpclassifier_roc.png', bbox_inches='tight')
# show roc curve
plt.show();