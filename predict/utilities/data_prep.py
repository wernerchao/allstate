import numpy as np
import pandas as pd

class data_prep(object):
    ''' A class for data preprocessing. '''

    @staticmethod
    def data_prep(train_data, target=True):
        ''' Data preprocessing to fill in missing values and one hot encoding.
        Return cleaned features & target.'''

        # Making columns for features
        cat_names = [i for i in train_data.columns if 'cat' in i]
        train_data = pd.get_dummies(data=train_data, columns=cat_names)
        features_mlp = [x for x in train_data.columns if x not in ['id', 'loss']]

        train_mlp_x = np.array(train_data[features_mlp])
        if target:
            train_mlp_y = np.array(train_data['loss'])
        else:
            train_mlp_y = None

        return train_mlp_x, train_mlp_y


    @staticmethod
    def data_prep_log(train_data, target=True):
        ''' Data preprocessing to fill in missing values and log the target variable.
        Return cleaned features & target.'''

        if target:
            train_xg['log_loss'] = np.log(train_xg['loss'])
            print 'train_xg shape:', train_xg.shape
            train_xg_y = np.array(train_xg['log_loss'])
        else:
            train_xg_y = None

        # Creating columns for features, and categorical features
        cat_features_col = [x for x in train_xg.select_dtypes(include=['object']).columns[0:131]]
        for c in range(len(cat_features_col)):
            train_xg[cat_features_col[c]] = train_xg[cat_features_col[c]].astype('category').cat.codes

        features_col = [x for x in train_xg.columns[0:131]]
        print 'features_col check 1: ', train_xg[features_col].shape
        train_xg_x = np.array(train_xg[features_col])
        

