# CodeRepo

This is my general workflow code repo for basic tasks, will update correspondingly. Lazy~~

### 01_ETL
### 02_EDA
### 03_FeatureSelection

### 04_FeaturePreprocessing

+ Handling Missing Value
  a. numeric: fill  with median(*mean or specific value)
  b. object: fill with 'Missing'
+ Numerci
  a. **Tree-based** models doesn't depend on scaling, while **non-tree-based** models hugely depend on scaling.
  b. MinMaxScaler to [0,1]
  b. StandardScaler to mean=0, std=1
  c. outlier clip
+ Categorical
  a. label encoding: alphabetical(sorted) vs order of apperance
  b. frequency encoding
  c. one-hot encoding
+ Text
+ Datetime

### 05_Modeling
+ Tree Based
  a. RandomForest
  b. GBDT(lightGBM)
+ Non-Tree Based
  a. Linear
  b. Neural Network
  
### 06_Evaluation
+ Classification 
  a. roc curve, auc score (only work for binary)
  b. precison & recall
  c. confusion matrix (redefine cost, fp&fn)
  d. decile report(KS, Gain&Lift Chart)
