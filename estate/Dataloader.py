import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from IPython.display import display, HTML
from sklearn import preprocessing

class Dataloader(object):
    def __init__(self):
        self.entity_features_columns = ['building_material', 'city', 'town', 'village', 'building_type', 'building_use', 'parking_way', 'I_index_50', 'I_index_500', 'I_index_1000', 'I_index_5000', 'I_index_10000', 'II_index_50', 'II_index_500', 'II_index_1000', 'II_index_5000', 'II_index_10000', 'III_index_50', 'III_index_500', 'III_index_1000', 'III_index_5000', 'III_index_10000', 'IV_index_50', 'IV_index_500', 'IV_index_1000', 'IV_index_5000', 'IV_index_10000', 'V_index_50', 'V_index_500', 'V_index_1000', 'V_index_5000', 'V_index_10000', 'VI_index_50', 'VI_index_500', 'VI_index_1000', 'VI_index_5000', 'VI_index_10000', 'VII_index_50', 'VII_index_500', 'VII_index_1000', 'VII_index_5000', 'VII_index_10000', 'VIII_index_50', 'VIII_index_500', 'VIII_index_1000', 'VIII_index_5000', 'VIII_index_10000', 'IX_index_50', 'IX_index_500', 'IX_index_1000', 'IX_index_5000', 'IX_index_10000', 'X_index_50', 'X_index_500', 'X_index_1000', 'X_index_5000', 'X_index_10000', 'XI_index_50', 'XI_index_500', 'XI_index_1000', 'XI_index_5000', 'XI_index_10000', 'XII_index_50', 'XII_index_500', 'XII_index_1000', 'XII_index_5000', 'XII_index_10000', 'XIII_index_50', 'XIII_index_500', 'XIII_index_1000', 'XIII_index_5000', 'XIII_index_10000', 'XIV_index_50', 'XIV_index_500', 'XIV_index_1000', 'XIV_index_5000', 'XIV_index_10000','parking_price_isna','txn_floor_isna']
    def prepare_train_data(self,df,one_hot = True,size=0.2):
        df['avg_price'] = df['total_price']/df['building_area']
        df = df.drop(['total_price'],axis=1)
        remove_outlier_df = self.remove_outlier(df)
        expend_feature_df = self.expend_feature(remove_outlier_df)
        df_fillna = self.fillna(expend_feature_df,method='regression')
        df_nor = self.normalize(df)
        train_y = df_nor['avg_price']
        train_x = df_nor.drop(['building_id','avg_price','parking_area'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=size, random_state=42)
        return X_train, X_test, y_train, y_test
    def prepare_test_data(self,df):
        expend_feature_df = self.expend_feature(df)
        df_fillna = self.fillna(expend_feature_df,method='regression')
        test_df = df_fillna.drop(['building_id','parking_area'],axis=1)
        return test_df
    def fillna(self,df,method = "zero"):
        if method == 'zero':
            return df.fillna(0)
        table = df['parking_price'].isna().values
        data = np.array([])
        y = np.array([])
        for index,t in enumerate(table):
            if not t:
                temp1 = df['parking_price'].iloc[index]
                temp2 = df['XIII_5000'].iloc[index]
                data = np.append(data,temp1)
                y = np.append(y,temp2)
        parking_price_reg = LinearRegression().fit(y.reshape(-1,1), data.reshape(-1,1))
        pre = parking_price_reg.predict(df['XIII_5000'].values.reshape(-1,1))
        table = pd.isnull(df['parking_price'])
        prediction = pre.reshape(-1,)
        col = np.array([])
        col2 = df['parking_price'].values
        for index,t in enumerate(table):
            if t:
                temp = prediction[index]
                col = np.append(col,temp)
            else:
                col = np.append(col,col2[index])
        df['parking_price'] = col
        
        df['txn_floor'] = df['txn_floor'].fillna(df['total_floor'])
        return df
            
    def expend_feature(self,X):
        na_features = ['parking_price','txn_floor']
        for na_feature in na_features:
            X[f"{na_feature}_isna"] = X.isnull()[na_feature]
        X['sell_day'] = X['txn_dt'] - X['building_complete_dt']
        return X
    def remove_outlier(self,X):
        outlier_index = np.where(np.abs(stats.zscore(X['avg_price'])) > 3)[0]
        clean_data = X.drop(X.index[outlier_index])
        return clean_data
    def normalize(self,X):
        for col in X.columns:
            if col not in entity_features_columns:
                X[col] = preprocessing.scale(X[col])
        return X