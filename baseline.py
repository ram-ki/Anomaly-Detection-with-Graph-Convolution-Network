import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score

class Model:

    def __init__(self, dataset = 'BlogCatalog', datadir = 'data'):

        data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
        adj = data_mat['Network']
        feat = data_mat['Attributes']
        truth = data_mat['Label']
        truth = truth.flatten()

        self.X = feat.toarray()
        self.y = truth

        print(self.X[0])
        print(self.y)

    def split_data(self):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)

    def scale_data(self):

        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
    
    def model_data(self):

        self.xgb_cl = xgb.XGBClassifier()
        self.xgb_cl.fit(self.X_train, self.y_train)
    
    def predict_data(self):

        self.preds = self.xgb_cl.predict(self.X_test)
        print('Auc', roc_auc_score(self.y_test, self.preds))
    
    def run(self):

        self.split_data()
        self.scale_data()
        self.model_data()
        self.predict_data()

if __name__ == '__main__':

    obj = Model()
    obj.run()





