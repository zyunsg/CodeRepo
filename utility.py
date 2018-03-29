# coding: utf-8
# file name: utility.py
# author: zyun
# date created: 3/20/2018
# date last modified: 3/29/2018
# python version: 2.7

import os
import pandas as pd
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


def load_csv(path, data):
    '''
    to load data
    path: the path where store data
    data: name of data, 'train.csv', 'test.csv', e.g.
    
    return:
    dataframe
    
    *example:
    data_path = '/sandbox/cbgba/Modeling/Users/zhangyun/CR/data/in/'
    df_train = load_data(data_path, 'TRAIN.csv')
    df_test = load_data(data_path, 'TEST.csv')
    
    '''
    csv_path = os.path.join(path, data)
    return pd.read_csv(csv_path)

def quick_look(data, label_f=[], keyid_f=[], datetime_f=[], coordinate_f=[]):
    
    '''
    to check data following information & construct feature list for future data processing
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
    dstruct = pd.DataFrame(data.dtypes).rename(columns={0:'dtype'})
    
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


def underSampling(data, label, k=5):
    '''
    underSampling for binary classification
    data: dataset
    label: target y
    k: undersampling ratio, k:1 (majority : minority)
    
    return 
    dataframe after undersampling by label
    
    example:
    train = underSampling(df_train, 'event', k=5)
    '''
    n_sample = data[label].value_counts()[1] * k 
    temp0 = data[data[label] == 0].sample(n=n_sample, random_state=1) 
    temp1 = data[data[label] == 1]
    return temp0.append(temp1)


def decilestats(true, pred, group=10):
    '''
    to generate the decile Stats(Ks, Gain, Lift@decile, Lift@total).
    Parameters:
    true: actual target values
    pred: prediction values
    group: number of deciles
    return:
    prob_min: minimum probablity
    prob_max: maximum probablity
    #pop: number of cases in its group
    #num1: number of positive(events/responses..)
    #num0: number of negative(non events/responses..)
    %pop: percentage of cases
    %num1: percentage of positive
    %num0: percentage of negative
    cum%pop: cumulative percentage of cases
    cum%num1: cumulative percentage of positive
    cum%num0: cumulative percentage of negative
    ks: the degree of separation between the positive and negative distributions
    lift@decile: lift in each group, (%num1/%pop)
    lift@total: lift total, (cum%num1/cum%pop)
    
    return: decile stats result
    #example:
    dstats = decilestats(y_true, y_pred)
    '''

    # dataframe:
    pdict = pd.DataFrame({"true": true, "pred": pred})
    pdict['decile'] = pd.qcut(pdict['pred'], group, duplicates='drop', labels=False)
    report = pdict.groupby(['decile']).agg({'true': {'#pop': 'count', '#num1': 'sum'},
                                            'pred': {'prob_min': 'min', 'prob_max': 'max'}})

    report.columns = report.columns.droplevel(0)  # drop level0('true', 'pred')
    report.sort_index(ascending=False, inplace=True)  # sort index

    # counts
    report['#num0'] = report['#pop'] - report['#num1']

    # percent
    report['%pop'] = report['#pop'] / sum(report['#pop'])
    report['%num1'] = report['#num1'] / sum(report['#num1'])
    report['%num0'] = report['#num0'] / sum(report['#num0'])

    # cumulative
    report['cum%pop'] = report['%pop'].cumsum()
    report['cum%num1'] = report['%num1'].cumsum()
    report['cum%num0'] = report['%num0'].cumsum()

    # result
    report['ks'] = report['cum%num1'] - report['cum%num0']
    report['lift@decile'] = report['%num1'] / report['%pop']
    report['lift@total'] = report['cum%num1'] / report['cum%pop']

    return report.round(3)


def decilegraph(df, ks=True, gain=True, lift=True):
    '''
    to generate decile Charts: KS, Gain, Lift@Decile, Lift@Total
    df: dataframe from DecileStats
    ks: flag for K-S Chart
    gain: flag for Gain Chart
    lift: flag for Lift Chart
    
    return:ks, gain & lift Charts
    '''

    # K-S Chart
    if ks == True:

        x1 = [0] + df['cum%pop'].tolist()
        y1 = [0] + df['cum%num1'].tolist()
        y2 = [0] + df['cum%num0'].tolist()

        trace1 = go.Scatter(name='%Event', x=x1, y=y1)
        trace2 = go.Scatter(name='%Non-event', x=x1, y=y2)
        layout = go.Layout(
            width=550, height=400,
            title='K-S Chart',
            xaxis=dict(title='%Population', range=[0, 1]),
            yaxis=dict(title='%Count')
        )
        data = [trace1, trace2]

        ks_graph = go.Figure(data=data, layout=layout)
    else:
        ks_graph = None

    # Gain Chart
    if gain == True:

        x1 = [0] + df['cum%pop'].tolist()
        y1 = [0] + df['cum%num1'].tolist()

        trace1 = go.Scatter(name='Model', x=x1, y=y1)
        trace2 = go.Scatter(name='Random', x=x1, y=x1)
        layout = go.Layout(
            width=550, height=400,
            title='Cummulative Gain Chart',
            xaxis=dict(title='%Population', range=[0, 1]),
            yaxis=dict(title='%Event')
        )
        data = [trace1, trace2]

        gain_graph = go.Figure(data=data, layout=layout)
    else:
        gain_graph = None

    # Lift Chart
    if lift == True:

        x1 = df['cum%pop'].tolist()
        y1 = df['lift@decile'].tolist()
        y2 = [1] * 10
        y3 = df['lift@total'].tolist()

        # Lift@Decile
        lift_d1 = go.Scatter(name='Model', x=x1, y=y1)
        lift_d2 = go.Scatter(name='Random', x=x1, y=y2)

        # Lift@Total
        lift_t1 = go.Scatter(name='Model', x=x1, y=y3)
        lift_t2 = go.Scatter(name='Random', x=x1, y=y2)

        lift_graph = plotly.tools.make_subplots(rows=1, cols=2,
                                                subplot_titles=(('Lift@Decile Chart', 'Lift@Total Chart')),
                                                print_grid=False)

        lift_graph.append_trace(lift_d1, 1, 1)
        lift_graph.append_trace(lift_d2, 1, 1)
        lift_graph.append_trace(lift_t1, 1, 2)
        lift_graph.append_trace(lift_t2, 1, 2)

        lift_graph['layout']['xaxis1'].update(title='%Population', showgrid=False)
        lift_graph['layout']['xaxis2'].update(title='%Population', showgrid=False)
        lift_graph['layout']['yaxis1'].update(title='Lift@decile', showgrid=False)
        lift_graph['layout']['yaxis2'].update(title='Lift@total', showgrid=False)
        lift_graph['layout'].update(title='Lift Chart')
    else:
        lift_graph = None

    return ks_graph, gain_graph, lift_graph

def decilereport(y_true, y_pred, group=10, ks_f=True, gain_f=True, lift_f=True):
    '''
    to generate the decile report, KS, Gain, Lift.
    y_true: true target
    y_pred: pred target
    group: number of groups, default=10
    ks_f: flag for ks chart
    gain_f: flag for gain chart
    lift_f: flg for lift chart
    return:
    s1: stats for decile report
    g1: K-S chart
    g2: Gain chart
    g3: Lift chart
    
    *example:
    dstats, ks_g, gain_g, lift_g = decilereport(y_true, y_pred)
    plotly.offline.iplot(ks_g, filename='ks_chart')
    plotly.offline.iplot(gain_g, filename='gain_chart')
    plotly.offline.iplot(lift_g, filename='lift_chart')
    '''
    s1 = decilestats(true=y_true, pred=y_pred, group=10)
    g1, g2, g3 = decilegraph(s1, ks=ks_f, gain=gain_f, lift=lift_f)

    return s1, g1, g2, g3

