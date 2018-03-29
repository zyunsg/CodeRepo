# Utility Functions 

### utility function list:
1. load_csv
2. quick_look
3. numStats
4. catVcount
5. underSampling
6. decilestats, decilegraph, decilereport

```
import utility as utl
```

+ **load_csv**: load csv dataset
```
from utility import load_csv

data_path = '../data/in/'
df_train = load_csv(data_path, 'train.csv')
df_test = load_csv(data_path, 'test.csv')
```
+ **quick_look**: quickly check following data information 
  + data shape
  + data preview
  + data types
  + missing value count and ratio
```
from utility import quick_look

train_shape, train_head, train_dstruct, train_dtype = quick_look(df_train, label_f=['event'], keyid_f=['PARTY_ID'])
test_shape, test_head, test_dstruct, test_dtype = quick_look(df_test, keyid_f=['PARTY_ID'])

num_f = train_dtype['num_f']
cat_f = train_dtype['cat_f']
key_f = train_dtype['key_f']
coord_f = train_dtype['coord_f']
dtime_f = train_dtype['dtime_f']
label_f = train_dtype['label_f']
```
+ **numStats**: stats description for numeric variables
```
from utility import numStats

num_f = ['a', 'A_6m_tamt_Giro', 'b', 'c']
numStats(df_train, num_f)
```
   
+ **catVcount**: check distribution for category variables 
```
from utility import catVcount

cat_f = ['a', 'b', 'c']
catVcount(data, cat_f[0])
```
+ **underSampling**: underSamping unbalanced dataset
```
from utility import underSampling

train = underSampling(df_train, 'event', k=5)
```
+ **decilereport**, **decilestats**, **decilegraph**: generate decile report
```
from utility import decilestats, decilegraph, decilereport

dstats = decilestats(y_true, y_pred)
dstats, ks_g, gain_g, lift_g = decilereport(y_true, y_pred)
plotly.offline.iplot(ks_g, filename='ks_chart')
plotly.offline.iplot(gain_g, filename='gain_chart')
plotly.offline.iplot(lift_g, filename='lift_chart')
```
