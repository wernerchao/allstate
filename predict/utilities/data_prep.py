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
        features_mlp = [x for x in train_data.columns if x not in ['id','loss']]

        train_mlp_x = np.array(train_data[features_mlp])
        if target:
            train_mlp_y = np.array(train_data['loss'])
        else:
            train_mlp_y = None

        return train_mlp_x, train_mlp_y
