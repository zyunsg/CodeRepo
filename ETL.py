# coding: utf-8
# file name: 01_ETL.py
# author: zyun
# date created: 3/20/2018
# date last modified: 3/23/2018
# python version: 2.7

import os
import pandas as pd


def load_csv(path, data):
    '''
    data loading
    path: the path where store data
    data: name of data, 'train.csv', 'test.csv', e.g.

    return:
    dataframe

    *example:
    data_path = '/sandbox/cbgba/Modeling/Users/zhangyun/CR/data/in/'
    df_train = load_csv(data_path, 'TRAIN.csv')
    df_test = load_csv(data_path, 'TEST.csv')

    '''
    csv_path = os.path.join(path, data)
    return pd.read_csv(csv_path)


def quick_look(data, label_f=[], keyid_f=[], datetime_f=[], coordinate_f=[]):
    '''
    data quick check & feature list construction
        a. data shape
        b. data types
        c. data preview
        d. misssing value count and ratio

    *note that need to mannually type in label, keyid, datetime, coordinate features list. default []

    *example
    train_shape, train_head, train_dstruct, train_dtype = quick_look(df_train, label_f=['event'], keyid_f=['PARTY_ID'])
    test_shape, test_head, test_dstruct, test_dtype = quick_look(df_test, keyid_f=['PARTY_ID'])

    num_f = train_dtype['num_f']
    cat_f = train_dtype['cat_f']
    key_f = train_dtype['key_f']
    coord_f = train_dtype['coord_f']
    dtime_f = train_dtype['dtime_f']
    label_f = train_dtype['label_f']
    '''

    #assertation
    assert type(label_f) == list
    assert type(keyid_f) == list
    assert type(datetime_f) == list
    assert type(coordinate_f) == list

    #basic: shape, head
    dshape = data.shape
    dhead = data.head()

    #data structure: data types
    dstruct = pd.DataFrame(data.dtypes).rename(columns={0: 'dtype'})

    #numeric & category features
    temp = set(label_f + keyid_f + datetime_f + coordinate_f)
    numeric_temp = set(dstruct[dstruct['dtype'] != object].index)
    category_temp = set(dstruct[dstruct['dtype'] == object].index)
    numeric_f = list(numeric_temp - temp)
    category_f = list(category_temp - temp)

    #data types
    dstruct['dtype2'] = None
    dstruct.loc[numeric_f, 'dtype2'] = 'numeric'
    dstruct.loc[category_f, 'dtype2'] = 'category'
    dstruct.loc[keyid_f, 'dtype2'] = 'key'
    dstruct.loc[datetime_f, 'dtype2'] = 'datetime'
    dstruct.loc[coordinate_f, 'dtype2'] = 'coordinate'
    dstruct.loc[label_f, 'dtype2'] = 'label'

    #missing value count / ratio
    mvc = pd.Series(data.isnull().sum(), name='mvcount')
    mvr = pd.Series(mvc / len(data), name='mvratio')
    dstruct = pd.concat([dstruct, mvc, mvr], axis=1)

    #data type list
    dtype = {
        'num_f': numeric_f,
        'cat_f': category_f,
        'key_f': keyid_f,
        'dtime_f': datetime_f,
        'coord_f': coordinate_f,
        'label_f': label_f
    }

    return dshape, dhead, dstruct, dtype


def numStats(data, numeric_f):
    '''
    stats for numeric data
    data: dataset
    numeric_f: numeric feature list

    *example
    num_f = ['a', 'A_6m_tamt_Giro', 'b', 'c']
    numStats(df_train, num_f)
    '''

    return data.loc[:, numeric_f].describe().T


def catVcount(data, colname):
    '''
    Value count for category features: count and ratio.
    data: dataset
    colname: category features name

    *example
    cat_f = ['a', 'b', 'c']
    catVcount(data, cat_f[0])
    '''

    vcount = pd.Series(data[colname].value_counts(dropna=False), name='count')
    vratio = pd.Series(data[colname].value_counts(normalize=True, dropna=False), name='ratio')
    return pd.concat([vcount, vratio], axis=1)
