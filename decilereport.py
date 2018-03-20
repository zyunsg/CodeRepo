import pandas as pd
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)

'''
name='decile report',
version='0.1',
description='decile report (decile stats, K-S, gain, and lift charts plotting on Jupyter)',
url='https://github.com/zyunsg/utilities/edit/master/decilereport.py',
author='Zhang Yun',
email='zyunsg@gmail.com'
date: 20/03/2018
'''


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
    '''
    s1 = decilestats(true=y_true, pred=y_pred, group=10)
    g1, g2, g3 = decilegraph(s1, ks=ks_f, gain=gain_f, lift=lift_f)

    return s1, g1, g2, g3
