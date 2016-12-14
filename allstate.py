import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# See the data first to get a feel. Seems like Test data is lacking 1 last col 'loss'
# print "Training data shape: ", train.shape, "; Testing data shape: ", test.shape
# print "\nFirst 20 cols (train): ", list(train.columns[:20]), "\n\n", "Last 20 cols (train): ", list(train.columns[-20:])
# print "\nFirst 20 cols (test): ", list(test.columns[:20]), "\n\n", "Last 20 cols (test): ", list(test.columns[-20:])
# print "\nDescribe train data: ", train.describe()
# print "\nAny missing data in train: ", pd.isnull(train).values.any(), "; Test: ", pd.isnull(test).values.any()
print "\nCheck categorical/continuous data: ", train.info()

# Confirming how many categorical features (object)
cat_features = list(train.select_dtypes(include=['object']).columns)
print "\nCategorical: {} features".format(len(cat_features))

# Confirming how many continuous features (float + int)
cont_features = []
for cont in list(train.select_dtypes(include=['float64', 'int64']).columns):
    if cont not in ['loss', 'id']:
        cont_features.append(cont)
print "\nContinuous: {} features".format(len(cont_features))

# Confirming id is the only leftover value
id_col = list(train.select_dtypes(include=['int64']).columns)
print "A column that's int64: {}".format(id_col)

# Visualize histogram
cat_uniques = []





