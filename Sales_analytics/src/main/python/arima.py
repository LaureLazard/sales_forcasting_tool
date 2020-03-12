from lib_interface import (sqrt, pd, showDf, plt, json,
                            mean_squared_error, warnings, 
                            datetime, itertools, SARIMAX, tsa, file_manip)

warnings.filterwarnings("ignore")


### Fonction ###
def prepareY(train, focused):
    y = train[['Date', focused]].copy()
    y = y.set_index('Date')
    return y[focused].resample('MS').mean()

def prepareDS(ds, var):
    ds = ds[['Date', var]].copy()
    ds = ds.set_index('Date')
    return ds[var].resample('MS').mean()

def check_prevision(train, p,d,q,P,D,Q, *args, **kwargs):
    focused = kwargs.get('focused', None)
    exog = kwargs.get('exog', None)
    if focused is None:
        focused="Weekly_Sales"
    y = prepareDS(train, focused)
    if exog is None:
        ex = None
    else:
        ex = prepareDS(train, exog)
    yhat = sarimax_fit(y[:'2012-01-01'],ex[:'2012-01-01'], p,d,q,P,D,Q)
    pred = yhat.get_forecast(steps=len(y['2012-01-01':]), exog=ex['2012-01-01':])
    # pred = yhat.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=False)
    prevision_ci = pred.conf_int()

    ax = y.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='Predicted', alpha=.7)

    ax.fill_between(prevision_ci.index,
                prevision_ci.iloc[:, 0],
                prevision_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel(focused)
    plt.legend()
    plt.show()

    y_forecasted = pred.predicted_mean
    y_truth = y['2012-01-01':]

    # Compute the mean square error
    rms = sqrt(mean_squared_error(y_forecasted, y_truth))
    print('Forecast Root Mean Squared Error {}'.format(round(rms, 2)))
    return yhat

def plot_prevision(train, test ,p,d,q,P,D,Q, *args, **kwargs):


    focused = kwargs.get('focused', None)
    exog = kwargs.get('exog', None)
    if focused is None:
        focused="Weekly_Sales"
    y = prepareDS(train, focused)
    if exog is None:
        ex = None
        tex = None
    else:
        ex = prepareDS(train, exog)
        tex = prepareDS(test, exog)

    yhat = sarimax_fit(y,ex, p,d,q,P,D,Q)
    prevision = yhat.get_forecast(steps=len(test.set_index('Date').resample('MS').mean()), exog=tex)
    pred_ci = prevision.conf_int()
    ax = y.plot(label='observed', figsize=(14, 7))
    prevision.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel(focused)
    plt.legend()
    plt.show()

def sarimax_fit(y, ex, p,d,q,P,D,Q):
    mod = SARIMAX(y,ex,p,d,q, P,D,Q, 12, False, False)
    return mod.fit()

### Fonction ###


def gridSearch(train, *args, **kwargs ):

    from time import process_time
    t1_start = process_time() 

    focused = kwargs.get('focused', None)
    exog = kwargs.get('exog', None)
    if focused is None: focused="Weekly_Sales"
    y = prepareDS(train, focused)
    if exog is None:ex = None
    else:ex = prepareDS(train, exog)
    max_param = file_manip.get_reg_config(only=["MAX_ARIMA"])

    p = d = q = range(0, int(max_param[0])+1)
    pdq = list(itertools.product(p, d, q))

    data=[]
    increment=0

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    for param in pdq:
        for s_param in seasonal_pdq:
            try:
                results = sarimax_fit(y,ex, param[0], param[1], param[2], 
                                        s_param[0], s_param[1], s_param[2])
                increment+=1
                item = {"id":increment, "param":"SARIMA"+str(param)+str(s_param), "AIC":results.aic}
                data.append(item)
            except:
                continue
    t1_stop = process_time() 
    # return t1_stop-t1_start
    file_manip.makejson(data)

def modelSave(results,sto, dep):
    nme = "models/"+str(sto) + "_" + str(dep) + "_AIC_"+ str(results.aic)
    results.save(nme+'.json')

