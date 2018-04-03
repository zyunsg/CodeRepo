# CodeRepo

This is my workflow code repo for some general tasks, will update correspondingly. Lazy~~

### 01_ETL
### 02_EDA
### 03_FeatureSelection

### 04_FeaturePreprocessing
+ Handling Missing Value
  1. numeric: fill  with median(*mean or specific value)
  2. object: fill with 'missing'
+ Numerci
  1. `Tree-based` models doesn't depend on scaling, while `non-tree-based` models hugely depend on scaling.
  2. MinMaxScaler to [0,1]
  3. StandardScaler to mean=0, std=1
  4. outlier clip
+ Categorical
  1. label encoding: alphabetical(sorted) vs order of apperance
  2. frequency encoding
  3. one-hot encoding
+ Text
+ Datetime

### 05_Modeling
+ Tree Based
  1. RandomForest
  2. GBDT(lightGBM)
+ Non-Tree Based
  1. Linear
  2. Neural Network
  
### 06_Evaluation
+ Classification 
  1. roc curve, auc score (only work for binary)
  2. precison & recall
  3. confusion matrix (redefine cost, fp&fn)
  4. decile report(KS, Gain&Lift Chart)
