# utilities
useful functions

1. **decilereport.py**: to generate **decile stats**, **K-S**, **gain**, and **lift charts**.
     '''
     import decilereport as dr
     
     dstats, ks_g, gain_g, lift_g = dr.decilereport(y_true, y_pred)
     dstats
     dr.plotly.offline.iplot(ks_g, filename='ks_chart')
     dr.plotly.offline.iplot(gain_g, filename='gain_chart')
     dr.plotly.offline.iplot(lift_g, filename='lift_chart')  
     '''
