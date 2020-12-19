from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import warnings

# ignore warnings
warnings.filterwarnings(action='ignore')

## Data Import
# read out file to variable
data = pd.read_csv('./clinicaldata/UsedCombined.txt', sep='\t')
# drop unwanted index and diagnosis columns
data = data.drop(['Unnamed: 0', 'Diagnosis'], axis=1)
# save raw data for later visualization
rawinput = data


## Data Preparation
# make sure strings are all in lowercase and remove whitespace
data = data.applymap(lambda s: s.lower() if type(s) == str else s)
data = data.applymap(lambda t: t.strip() if type(t) == str else t)
# define list that will be removed later on
removeList = ['Sex', 'neutrophilCategorical', 'serumLevelsOfWhiteBloodCellCategorical', 'lymphocytesCategorical',
              'CTscanResults', 'XrayResults', 'Diarrhea', 'Fever', 'Coughing', 'SoreThroat', 'NauseaVomitting',
              'Fatigue', 'RenalDisease', 'diabetes']
# transform categorical variables into 0 or 1 and add to dataframe
for i in removeList:
    data = pd.concat([data, pd.get_dummies(data[i], prefix=i)], axis=1)
# drop list because we have dummies now
data = data.drop(columns=removeList)

## Missing Data Imputation
# impute missing (NaN) values using sklearns IterativeImputer
# initiate Imputer
imp = IterativeImputer(max_iter=10, random_state=1)
# fit the existing data
imp.fit(data)
# impute data and save in new variable
imputed_data = pd.DataFrame(imp.transform(data), columns=data.columns)
# use location selection to define X = features and y = D (diagnosis)
X, y = imputed_data.loc[:, data.columns != 'D'], data.loc[:, 'D']