# CodeRepo

This is my workflow code repo for some general tasks, will update correspondingly. Lazy~~

### 01_ETL
### 02_EDA
+ Quickly Check: data shape, types, etc.
+ Numeric
  + stats
+ Categorical 
  + category count/ratio
  
### 03_FeatureSelection
+ Iterative RandomForest feature importance selection

### 04_FeaturePreprocessing
+ Handling Missing Value
  + numeric: fill  with median(*mean or specific value)
  + object: fill with 'missing'
+ Numeric
  + `Tree-based` models doesn't depend on scaling, while `non-tree-based` models hugely depend on scaling.
  + MinMaxScaler to [0,1]
  + StandardScaler to mean=0, std=1
  + outlier clip
+ Categorical
  + label encoding: alphabetical(sorted) vs order of apperance
  + frequency encoding
  + one-hot encoding
+ Text
+ Datetime

### 05_Modeling
+ Tree Based
  + RandomForest
  + GBDT(lightGBM)
+ Non-Tree Based
  + Linear
  + Neural Network
  
### 06_Evaluation
+ Classification 
  + roc curve, auc score (only work for binary)
  + precison & recall
  + confusion matrix (redefine cost, fp&fn)
  + decile report(KS, Gain&Lift Chart)
