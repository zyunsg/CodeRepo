# Utility Functions


### decilereport.py 
to generate **decile stats**, **K-S**, **Gain**, and **Lift Charts**.
+ decile stats only:
    ```
     import decilereport as dr
     dstats = dr.decilestats(y_true, y_pred)
    ```
+ decile stats & charts:
    ```
     import decilereport as dr
     dstats, ks_g, gain_g, lift_g = dr.decilereport(y_true, y_pred)
     dstats
     dr.plotly.offline.iplot(ks_g, filename='ks_chart')
     dr.plotly.offline.iplot(gain_g, filename='gain_chart')
     dr.plotly.offline.iplot(lift_g, filename='lift_chart')  
    ```
