import pandas as pd

train = pd.read_csv("../ProcessedData/train.csv")

targetCol = train[["SalePrice"]]
for col in list(train.columns):
    predict = train[["SalePrice", col]]
    print(predict.corr())