import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import missingno as msno
from data_preparation import rawinput

## Raw Data Description
# show raw data shape
print('\nImported Data has the following shape: \n{} rows and {} columns'.format(len(rawinput.index),
                                                                                 len(rawinput.columns)))
# show all column names
print('Dataset contains the following columns:\n{}'.format(list(rawinput)))
# show basic description
print('\nBasic data summary:\n{}'.format(rawinput.describe()))
# show NaN values per column
print('Number of NaN values per column:\n{}'.format(rawinput.isna().sum()))

## Plot correlation matrix of features
# init matrix
msno.heatmap(rawinput, figsize=(9, 7), fontsize=5, labels=True)
# add title
plt.title('Correlation Matrix of dataset features')
# tight layout
plt.tight_layout()
# save as png file, remove all of whitespace around figure
plt.savefig('output/feature_correlation_matrix.png', bbox_inches='tight')
# show plot
plt.show();


## plot distribution of existent and missing values
# init missingno matrix
msno.matrix(rawinput, figsize=(9, 7), fontsize=4, labels=True)
# add title and labels
plt.title('Features and corresponding values')
plt.xlabel('Features')
plt.ylabel('Dataset Location')
# add labels and legend
grey_patch = mpatches.Patch(color='grey', label='Given data')
white_patch = mpatches.Patch(color='white', label='Missing Data')
plt.legend(handles=[white_patch, grey_patch], loc='upper left')
# tight layout
plt.tight_layout()
# save as png file, remove all of whitespace around figure
plt.savefig('output/NaN_distribution.png', bbox_inches='tight')
# show plot
plt.show();