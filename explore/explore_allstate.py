# This module runs a exploratory data analysis.
# Run this module first to visualize the data insight.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pylab

from scipy import stats
from copy import deepcopy

from sklearn.decomposition import PCA

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# See the data first to get a feel. Seems like Test data is lacking 1 last col 'loss'
print train.head(10)
print "Training data shape: ", train.shape, "; Testing data shape: ", test.shape
print "\nFirst 20 cols (train): ", list(train.columns[:20]), "\n\n", "Last 20 cols (train): ", list(train.columns[-20:])
print "\nFirst 20 cols (test): ", list(test.columns[:20]), "\n\n", "Last 20 cols (test): ", list(test.columns[-20:])
print "\nDescribe train data: ", train.describe()
print "\nAny missing data in train: ", pd.isnull(train).values.any(), "; Test: ", pd.isnull(test).values.any()
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
print "\nA column that's int64: {}".format(id_col)


# Visualizing training set Loss(Output) vs ID
fig_1 = plt.figure(figsize=(16, 8))
plt.plot(train['id'], train['loss'])
plt.title('Loss per ID')
plt.xlabel('ID')
plt.ylabel('Loss Value')
plt.legend()

# Checking the skewness of the data
ori_skewness = stats.mstats.skew(train['loss']).data
log_skewness = stats.mstats.skew(np.log(train['loss'])).data
print "Original data skewness: ", ori_skewness, "\nLogged data skewness: ", log_skewness

# Visualize histogram of skewness(data) and log(data)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.hist(train['loss'], bins=50)
ax1.set_title("Original Loss(train) Histogram")
ax2.hist(np.log(train['loss']), bins=50)
ax2.set_title("Logged Loss(train) Histogram")

# Visualize all histogram of all continusou features
# We can see some features have a lot of spikes, could be transformed from categorical to continuous
train[cont_features].hist(bins=50, figsize=(16, 12))
plt.show()
raw_input()


# Calculate Pearson correlation btwn all continuous features
# We see some features are highly correlated, i.e. cont1 & cont9
correlation_mat = train[cont_features].corr()
print correlation_mat


# Comparing training set and test set. See if they are distributed the same way
train_d = train.drop(['id', 'loss'], axis=1)
test_d = test.drop(['id'], axis=1)
# Add a new col to make sure we can distinguish them later
train_d['Target'] = 1
test_d['Target'] = 0
# Make a huge data set
data_d_all = pd.concat((train_d, test_d))
data_label = deepcopy(data_d_all)
# Transforming all alphabetical data into categorical numerics
for c in range(len(cat_features)):
    data_label[cat_features[c]] = data_label[cat_features[c]].astype('category').cat.codes
print "After cat.codes: ", data_label.head(10)


# Recreate training and test set
data_label = data_label.iloc[np.random.permutation(len(data_label))]
x = data_label.iloc[:, :130]
y = data_label.iloc[:, 130:]

pca = PCA(n_components=2)
x_trans = pca.fit_transform(x)

plt.figure(figsize=(16, 8))
plt.scatter(x_trans[:, 0], x_trans[:, 1], c=np.array(y), 
            edgecolor='none', s=40,
            cmap=plt.cm.get_cmap('winter', 2))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar()
plt.show()
