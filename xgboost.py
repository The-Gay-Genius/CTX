import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

df = pd.read_csv(r'CTX_data.csv')
df

x = df.drop(columns='CTX removal efficiency')
y = df['CTX removal efficiency']

n_splits = 10
KFold = KFold(n_splits, shuffle=False, random_state=None)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=114)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

params = {
    "n_estimators": 450,
    "max_depth": 5,
    "min_samples_split": 5,
    "learning_rate": 0.025,
}
model = ensemble.GradientBoostingRegressor(**params)

param_test_max_depth = {
 'max_depth':list(range(2,10,1))
}
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch1.fit(X_train,y_train)



param_test_max_depth = {
 'min_samples_split':list(range(2,10,1))
}
gsearch2 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch2.fit(X_train,y_train)


params = {
    "n_estimators": 450,
    "max_depth": 5,
    "min_samples_split": 3,
    "learning_rate": 0.025,
}
model = ensemble.GradientBoostingRegressor(**params)



param_test_max_depth = {
 'learning_rate':[i/100.0 for i in range(0,15,1)]
}
gsearch3 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch3.fit(X_train,y_train)


params = {
    "n_estimators": 450,
    "max_depth": 5,
    "min_samples_split": 3,
    "learning_rate": 0.04,
}
model = ensemble.GradientBoostingRegressor(**params)



param_test_max_depth = {
  'n_estimators':list(range(100,400,20))
}
gsearch3 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch3.fit(X_train,y_train)


params = {
    "n_estimators": 380,
    "max_depth": 5,
    "min_samples_split": 3,
    "learning_rate": 0.03,
}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(X_train, y_train)

y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse_tr_pls = mean_squared_error(y_train,y_train_pred,squared = False)
rmse_te_pls = mean_squared_error(y_test,y_test_pred,squared = False)

print('RMSE(traing)%.3f' % rmse_tr_pls)
print('RMSE(test)%.3f' % rmse_te_pls)

y_pred_train_gbr = model.predict(X_train)
y_pred_test_gbr = model.predict(X_test)
plt.figure(figsize = (4,4))
plt.scatter(y_train,y_pred_train_gbr,alpha = 0.5,color = 'blue',label = 'training')
plt.scatter(y_test,y_pred_test_gbr,alpha = 0.5,color = 'red',label = 'test')
plt.legend()
plt.xlabel('DFT')
plt.ylabel('predication')
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))

dt = pd.DataFrame(y_pred_train_gbr)
dt.to_csv("y_pred_train_gbr.csv", index=0)
dt = pd.DataFrame(y_pred_test_gbr)
dt.to_csv("y_pred_test_gbr.csv", index=0)
dt = pd.DataFrame(y_train)
dt.to_csv("y_train.csv", index=0)
dt = pd.DataFrame(y_test)
dt.to_csv("y_test.csv", index=0)