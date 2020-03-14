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
        self.entity_features_columns = ['building_material', 'city_town','building_type', 'building_use', 'parking_way','I_index','II_index','III_index','IV_index','V_index','VI_index','VII_index','VIII_index','IX_index','X_index','XI_index','XII_index','XIII_index','XIV_index' ,'parking_price_isna','txn_floor_isna']
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

    def train_LGBM(self,train, t_target, valid, v_target,parm,use_custom_loss = False,reg_alpha = 0,reg_lambda = 0):
        if use_custom_loss:
            self.loss = custom_loss
        learning_rate = parm['learning_rate']
        n_estimators = parm['n_estimators']
        max_depth = parm['max_depth']
        num_leaves = parm['num_leaves']
        feature_fraction = parm['feature_fraction']
        flag = True
        good_depth = 0
        good_leaves = 0
        good_fraction = 0
        
        for depth in max_depth:
            for leaves in num_leaves:
                for fraction in feature_fraction:
                    rf = LGBMRegressor(learning_rate=learning_rate, 
                                       objective='regression', 
                                       n_estimators=n_estimators,
                                       max_depth=depth, 
                                       num_leaves=leaves, 
                                       reg_alpha=reg_alpha,
                                       reg_lambda = reg_lambda,
                                       feature_fraction=fraction, 
                                       bagging_freq=1,
                                       metric='rmse')           
                    rf.fit(train, t_target, # should we drop the features that are not correlate to our target?
                           eval_set=[(train, t_target), (valid, v_target)],
                           verbose=5000,
                           eval_metric=self.loss,
                           categorical_feature=self.entity_features_columns
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
                        good_depth = depth
                        good_leaves = leaves
                        good_fraction = fraction
        return self
    def predict(self,X_test,y_test):
        yhat = self.model.predict(X_test)
        return yhat*X_test['building_area']  ,y_test*X_test['building_area']
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
