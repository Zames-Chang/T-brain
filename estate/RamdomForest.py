import pandas as pd 
import numpy as np 
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import display, HTML

class RamdomForest(object):
    def __init__(self):
        self.model = ""
        self.loss = 'l1'
        self.max_point = 0
    def score(self,y_true,y_predict):
        z = 0
        z2 = 0
        total = 0
        for y1,y2 in zip(y_predict,y_true):
            present = abs(y1-y2)/y2
            total += present
            if(present <= 0.1):
                z += 1
            if(present > 0.2):
                z2 += 1
        return z/len(y_true)*10000 + (1-total/len(y_true))
    def custom_loss(self,y_true, y_pred):
        z , point = score(y_true,y_pred)
        return 'custom_loss',z, False

    def train_LGBM(self,train, t_target, valid, v_target,parm,use_custom_loss = False):
        entity_features_columns = ['building_material', 'city', 'town', 'village', 'building_type', 'building_use', 'parking_way', 'I_index_50', 'I_index_500', 'I_index_1000', 'I_index_5000', 'I_index_10000', 'II_index_50', 'II_index_500', 'II_index_1000', 'II_index_5000', 'II_index_10000', 'III_index_50', 'III_index_500', 'III_index_1000', 'III_index_5000', 'III_index_10000', 'IV_index_50', 'IV_index_500', 'IV_index_1000', 'IV_index_5000', 'IV_index_10000', 'V_index_50', 'V_index_500', 'V_index_1000', 'V_index_5000', 'V_index_10000', 'VI_index_50', 'VI_index_500', 'VI_index_1000', 'VI_index_5000', 'VI_index_10000', 'VII_index_50', 'VII_index_500', 'VII_index_1000', 'VII_index_5000', 'VII_index_10000', 'VIII_index_50', 'VIII_index_500', 'VIII_index_1000', 'VIII_index_5000', 'VIII_index_10000', 'IX_index_50', 'IX_index_500', 'IX_index_1000', 'IX_index_5000', 'IX_index_10000', 'X_index_50', 'X_index_500', 'X_index_1000', 'X_index_5000', 'X_index_10000', 'XI_index_50', 'XI_index_500', 'XI_index_1000', 'XI_index_5000', 'XI_index_10000', 'XII_index_50', 'XII_index_500', 'XII_index_1000', 'XII_index_5000', 'XII_index_10000', 'XIII_index_50', 'XIII_index_500', 'XIII_index_1000', 'XIII_index_5000', 'XIII_index_10000', 'XIV_index_50', 'XIV_index_500', 'XIV_index_1000', 'XIV_index_5000', 'XIV_index_10000','parking_price_isna','txn_floor_isna']
        #entity_features_columns = ['building_material', 'city', 'town', 'village', 'building_type', 'building_use', 'parking_way','parking_price_isna','txn_floor_isna']
        if use_custom_loss:
            self.loss = custom_loss
        learning_rate = parm['learning_rate']
        n_estimators = parm['n_estimators']
        max_depth = parm['max_depth']
        num_leaves = parm['num_leaves']
        feature_fraction = parm['feature_fraction']
        flag = True
        
        for depth in max_depth:
            for leaves in num_leaves:
                for fraction in feature_fraction:
                    rf = LGBMRegressor(learning_rate=learning_rate, objective='regression', n_estimators=n_estimators,
                                       max_depth=depth, num_leaves=leaves,
                                       feature_fraction=fraction, bagging_freq=1,metric='rmse')           
                    rf.fit(train, t_target, # should we drop the features that are not correlate to our target?
                           eval_set=[(train, t_target), (valid, v_target)],
                           early_stopping_rounds=50, verbose=10,
                           eval_metric=self.loss,
                           categorical_feature=entity_features_columns
                           )
                    print("Finished.")
                    if flag:
                        self.model = rf
                        flag = False
                    y_predict ,y_true= self.predict(valid,v_target)
                    point = self.score(y_true,y_predict)
                    if point > self.max_point:
                        self.max_point = point
                        self.model = rf
        return self
    def predict(self,X_test,y_test):
        yhat = self.model.predict(X_test)
        return yhat*X_test['building_area'],y_test*X_test['building_area']
    def plot_feature_important(self):
        ax = lgb.plot_importance(self.model, max_num_features=20)
        return ax
    def plot_loss(self):
        if self.loss != 'l1':
            loss = 'custom_loss'
        else:
            loss = 'l1'
        loss1 = self.model.evals_result_['training'][loss]
        loss2 = self.model.evals_result_['valid_1'][loss]
        plt.title('blue : train, red : test')
        plt.plot(loss1,label='training')
        plt.plot(loss2,color='red',label='test')
        plt.show()