import pandas as pd 
import numpy as np 
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split
from Dataloader import Dataloader
from RamdomForest import RamdomForest
from score import score
from sklearn.utils import shuffle
from datetime import date

df = pd.read_csv('data/train.csv')
parm = {
    'learning_rate' : 0.1,
    'n_estimators' : 100000,
    'max_depth' : [64],
    'num_leaves' : [256],
    'feature_fraction' : [0.9]
}

dataloader = Dataloader()
X_train, X_test, y_train, y_test = dataloader.prepare_train_data(df)

ramdom_forest = RamdomForest()
model = ramdom_forest.train_LGBM(X_train, y_train, X_test, y_test,parm)

y, y_true= model.predict(X_test,y_test)
hit_rate,point = score(y,y_true)

test_df = pd.read_csv("data/test.csv")
build_id = pd.read_csv("data/submit_test.csv")['building_id']
test_df = dataloader.prepare_test_data(test_df)
test_y, _= model.predict(test_df,y_test)
submit_df = pd.DataFrame(data={
    'building_id' : build_id,
    'total_price' : test_y
})
today = str(date.today())
submit_df.to_csv(f'submits/{today}_lightgbm.csv',index = False)

print(f"hit_rate : {hit_rate}")
print(f"score : {point}")
