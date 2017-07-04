import pandas as pd
import math
from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

alpha_ridge = np.linspace(0.00001, 3, 5000)

testPath = '../ProcessedData/test.csv'
trainPath = '../ProcessedData/train.csv'

train = pd.read_csv(trainPath, header=0, index_col=0)
train = train[(train.LotArea < 100000) & (train.GrLivArea < 4000)]
min_price = min(train.SalePrice)
X_test = pd.read_csv(testPath, header=0, index_col=0)

y_train = train['SalePrice']
X_train = train.drop(['SalePrice'], axis=1)

for col in X_test.columns.values:
    for row in X_test[col]:
        if (math.isnan(row)):
            print("Error: " + str(col) + "---" + str(row))

lasso_reg = LassoCV(alphas=alpha_ridge)
lasso_reg.fit(X_train, y_train)
y_pred_train = pd.DataFrame(lasso_reg.predict(X_train)) 
mse = mean_squared_error(y_train, y_pred_train)

y_pred_test = lasso_reg.predict(X_test)

idRange = range(1461, 2920)
y_pred_test_export = pd.DataFrame()
y_pred_test_export['Id'] = idRange
y_pred_test_export['SalePrice'] = y_pred_test
y_pred_test_export.columns = ["Id", "SalePrice"]
y_pred_test_export.ix[y_pred_test_export["SalePrice"] < min_price, "SalePrice"] = min_price
y_pred_test_export.to_csv('../Predictions/predictions_lasso.csv', index=False)