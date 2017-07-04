import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

num_features = range(3, 71, 1)

testPath = '../ProcessedData/test_extra.csv'
trainPath = '../ProcessedData/train_extra.csv'

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

regression_scores = []
            
for n in num_features:
    trial_forest = RandomForestRegressor(max_features=n, oob_score=True)
    trial_forest.fit(X_train, y_train)
    regression_scores.append(trial_forest.oob_score_)

plt.scatter(num_features, regression_scores)
plt.show()

best_feature_quantity = regression_scores.index(min(regression_scores)) + 3       
forest = RandomForestRegressor(max_features=best_feature_quantity)
forest.fit(X_train, y_train)
y_pred_train = pd.DataFrame(forest.predict(X_train)) 
mse = mean_squared_error(y_train, y_pred_train)

y_pred_test = forest.predict(X_test)

idRange = range(1461, 2920)
y_pred_test_export = pd.DataFrame()
y_pred_test_export['Id'] = idRange
y_pred_test_export['SalePrice'] = y_pred_test
y_pred_test_export.columns = ["Id", "SalePrice"]
y_pred_test_export.ix[y_pred_test_export["SalePrice"] < min_price, "SalePrice"] = min_price
y_pred_test_export.to_csv('../Predictions/predictions_forest.csv', index=False)