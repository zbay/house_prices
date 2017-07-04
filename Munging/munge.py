import pandas as pd

train = pd.read_csv("../RawData/train.csv", index_col='Id', header=0)
test = pd.read_csv("../RawData/test.csv", index_col='Id', header=0)
train.head()

def processData(inputURL, outputURL, outputExtraURL):
    frame = pd.read_csv(inputURL, index_col='Id', header=0)
    print(frame.info())
    frame["LotFrontage"] = frame["LotFrontage"].fillna(0.0)
    frame["MasVnrArea"] = frame["MasVnrArea"].fillna(0.0)
    frame["GarageYrBlt"] = frame["GarageYrBlt"].fillna(1930.0)
    frame["YrSold"] = frame["YrSold"] + ((frame["MoSold"]-1)/12.0)
    
    del frame["MoSold"]
    del frame["PoolQC"]
    del frame["Street"]
    del frame["Heating"]
    del frame["Utilities"]
    del frame["MiscFeature"]
    del frame["MasVnrType"]
    
    frame["PorchArea"] = frame["WoodDeckSF"] + frame["OpenPorchSF"] + frame["EnclosedPorch"] + frame["3SsnPorch"] + frame["ScreenPorch"]
    del frame["WoodDeckSF"]
    del frame["OpenPorchSF"]
    del frame["EnclosedPorch"]
    del frame["3SsnPorch"]
    del frame["ScreenPorch"]
    
    booleanMap = {"True": 1.0, "TRUE": 1.0, True: 1.0, "FALSE": 0.0, "False": 0.0, False: 0.0}
    frame["WoodFence"] = ((frame["Fence"] == "MnWw") | (frame["Fence"] == "GdWo")).map(booleanMap)
    frame["PrivacyFence"] = ((frame["Fence"] == "MnPrv") | (frame["Fence"] == "GdPrv")).map(booleanMap)
    del frame["Fence"]
    frame["OnAlley"] = ((frame["Alley"] == "Grvl") | (frame["Alley"] == "Pave")).map(booleanMap)
    del frame["Alley"]
    frame["LandSlopeGentle"] = (frame["LandSlope"] == "Gtl").map(booleanMap)
    del frame["LandSlope"]
    frame["StandardShingled"] = (frame["RoofMatl"] == "CompShg").map(booleanMap)
    del frame["RoofMatl"]
    
    frame["CentralAir"] = (frame["CentralAir"] == "Y").map(booleanMap)
    frame["PavedDrive"] = (frame["PavedDrive"] == "Y").map(booleanMap)
    
    generalMap = {"Ex": 6.0, "Gd": 4.0, "TA": 3.0, "Ta": 3.0, "Fa": 2.0, "Po": 1.0, "NA": 0.0} # for kitchen, exterior, and heating
    basementExposureMap = {"NA": 0.0, "No": 1.0, "Mn": 3.0, "Avg": 5.0, "Gd": 7.0}
    functionalMap = {"Sal": 0.0, "Sev": 1.0, "Maj2": 2.0, "Maj1": 3.0, "Mod": 5.0, "Min2": 7.0, "Min1": 9.0, "Typ": 12.0, "NA": 1.0}
    garageFinishMap = {"NA": 0.0, "Unf": 1.0, "RFn": 2.0, "Fin": 4.0}
    electricalMap = {"FuseP": 0.0, "FuseF": 1.0, "FuseA": 2.0, "SBrkr": 3.0, "Mixed": 1.5, "NA": 0.0}
    
    frame["KitchenQual"] = frame["KitchenQual"].map(generalMap)
    frame["ExterQual"] = frame["ExterQual"].map(generalMap)
    frame["ExterCond"] = frame["ExterCond"].map(generalMap)
    frame["HeatingQC"] = frame["HeatingQC"].map(generalMap)
    frame["GarageQual"] = frame["GarageQual"].map(generalMap)
    frame["GarageCond"] = frame["GarageCond"].map(generalMap)
    frame["FireplaceQu"] = frame["FireplaceQu"].map(generalMap)
    frame['BsmtQual'] = frame['BsmtQual'].map(generalMap)
    frame['BsmtCond'] = frame['BsmtCond'].map(generalMap)
    frame['BsmtExposure'] = frame['BsmtExposure'].map(basementExposureMap)
    frame["Functional"] = frame["Functional"].map(functionalMap)
    frame["GarageFinish"] = frame["GarageFinish"].map(garageFinishMap)
    frame["Electrical"] = frame["Electrical"].map(electricalMap)
    
    basementFinishMap = {"GLQ": 6.0, "ALQ": 5.0, "BLQ": 4.0, "Rec": 3.0, "LwQ": 2.0, "Unf": 1.0, "NA": 0.0}
    # for basement, do a weighted average involving the sq footage of each area
    bsmt_proportion_1 = frame["BsmtFinSF1"] / frame['TotalBsmtSF']
    bsmt_proportion_2 = frame["BsmtFinSF2"] / frame['TotalBsmtSF']
    bsmt_quality_1 = frame["BsmtFinType1"].map(basementFinishMap)
    bsmt_quality_2 = frame["BsmtFinType2"].map(basementFinishMap)
    frame["BsmtFinQualAvg"] = (bsmt_quality_1 * bsmt_proportion_1) + (bsmt_quality_2 * bsmt_proportion_2)
    del frame["BsmtFinSF1"]
    del frame["BsmtFinSF2"]
    del frame["BsmtFinType1"]
    del frame["BsmtFinType2"]
    
    transportationOne = (frame["Condition1"] != "Norm").map(booleanMap)
    transportationTwo = (frame["Condition1"] != "Norm").map(booleanMap)
    frame["TransportHub"] = transportationOne + transportationTwo
    del frame["Condition1"]
    del frame["Condition2"]
    
    frame["Toilets"] = frame["FullBath"] + frame["HalfBath"]
    frame["Showers"] = frame["FullBath"]
    del frame["FullBath"]
    del frame["HalfBath"]
    
    frame["VinylSiding"] = ((frame["Exterior1st"] == 'VinylSd') | (frame["Exterior2nd"] == 'VinylSd')).map(booleanMap)
    frame["MetalSiding"] = ((frame["Exterior1st"] == 'MetalSd') | (frame["Exterior2nd"] == 'MetalSd')).map(booleanMap)
    frame["WoodSiding"] = ((frame["Exterior1st"] == 'Wd Sdng') | (frame["Exterior2nd"] == 'Wd Sdng')).map(booleanMap)
    frame["BrickFace"] = ((frame["Exterior1st"] == 'BrkFace') | (frame["Exterior2nd"] == 'BrkFace')).map(booleanMap)
    frame["BrickCommon"] = ((frame["Exterior1st"] == 'BrkComm') | (frame["Exterior2nd"] == 'BrkCmn') | (frame["Exterior2nd"] == 'BrkComm')).map(booleanMap)
    frame["PlywoodExterior"] = ((frame["Exterior1st"] == 'Plywood') | (frame["Exterior2nd"] == 'Plywood')).map(booleanMap)
    frame["StuccoExterior"] = ((frame["Exterior1st"] == 'Stucco') | (frame["Exterior2nd"] == 'Stucco')).map(booleanMap)
    frame["AsbestosShingles"] = ((frame["Exterior1st"] == 'AsbShng') | (frame["Exterior2nd"] == 'AsbShng')).map(booleanMap)
    frame["WoodShingles"] = ((frame["Exterior1st"] == 'WdShing') | (frame["Exterior2nd"] == 'Wd Shng')).map(booleanMap)
    frame["CementBoard"] = ((frame["Exterior1st"] == 'CemntBd') | (frame["Exterior2nd"] == 'CemntBd')).map(booleanMap)
    frame["HardBoard"] = ((frame["Exterior1st"] == 'HdBoard') | (frame["Exterior2nd"] == 'HdBoard')).map(booleanMap)
    frame["ImitationStucco"] = ((frame["Exterior1st"] == 'ImStucc') | (frame["Exterior2nd"] == 'ImStucc')).map(booleanMap)
    
    del frame["Exterior1st"]
    del frame["Exterior2nd"]
    
    frame["BsmtExposure"] = frame["BsmtExposure"].fillna(0.0)
    frame["BsmtQual"] = frame["BsmtQual"].fillna(0.0)
    frame["BsmtCond"] = frame["BsmtCond"].fillna(0.0)
    frame["FireplaceQu"] = frame["FireplaceQu"].fillna(0.0)
    frame["GarageFinish"] = frame["GarageFinish"].fillna(0.0)
    frame["Electrical"] = frame["Electrical"].fillna(2.0)
    frame["GarageQual"] = frame["GarageQual"].fillna(0.0)
    frame["GarageCond"] = frame["GarageCond"].fillna(0.0)
    frame["BsmtUnfSF"] = frame["BsmtUnfSF"].fillna(0.0)
    frame["TotalBsmtSF"] = frame["TotalBsmtSF"].fillna(0.0)
    frame["BsmtFullBath"] = frame["BsmtFullBath"].fillna(0.0)
    frame["KitchenQual"] = frame["KitchenQual"].fillna(0.0)
    frame["Functional"] = frame["Functional"].fillna(0.0)
    frame["GarageCars"] = frame["GarageCars"].fillna(0.0)
    frame["BsmtFinQualAvg"] = frame["BsmtFinQualAvg"].fillna(0.0)
    frame["BsmtHalfBath"] = frame["BsmtHalfBath"].fillna(0.0)
    frame["GarageArea"] = frame["GarageArea"].fillna(0.0) 
    
    frame = pd.get_dummies(frame, columns=['MSZoning',
        'LotShape', 'LandContour',
        'LotConfig', 'Neighborhood',
        'BldgType', 'HouseStyle',
        'RoofStyle', 'Foundation', 'GarageType',
        "SaleType", "SaleCondition", "MSSubClass"])
    
    frame.to_csv(outputURL, index=False)
    
    important_vars = ["GarageType_Detchd", "Foundation_CBlock", "MSZoning_RM", "LotShape_Reg", "SaleType_WD", "MSSubClass_30",
                 "RoofStyle_Gable", "Foundation_BrkTil", "Neighborhood_OldTown", "Neighborhood_Edwards", 
                  "OverallQual", "GrLivArea", "ExterQual", "KitchenQual", "GarageCars", "TotalBsmtSF", "BsmtQual",
                  "GarageArea", "1stFlrSF", "Toilets", "YearBuilt", "TotRmsAbvGrd"]
    for variable in important_vars:
        frame[variable + "_Squared"] = frame[variable] ** 2
        frame[variable + "_Cubed"] = frame[variable] ** 3
        frame[variable + "_SquareRoot"] = frame[variable] ** 0.5
    
    frame.to_csv(outputExtraURL, index=False)
    


processData("../RawData/train.csv", "../ProcessedData/train.csv", "../ProcessedData/train_extra.csv")
processData("../RawData/test.csv", "../ProcessedData/test.csv", "../ProcessedData/test_extra.csv")