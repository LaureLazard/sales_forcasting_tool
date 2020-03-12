from lib_interface import (sqrt, np, plt,
                            mean_squared_error, scale, 
                            datetime, LinearRegression, style)
style.use('ggplot')


def plotforecast(train, test):
    traindt = train.set_index('Date')
    testdt = test.set_index('Date')

    testdt.fillna(value=000, inplace=True)
    forecast_col='Weekly_Sales'

    X = np.array(traindt.drop(columns=[forecast_col]))
    X = scale(X)
    newX = np.array(testdt)
    newX = scale(newX)

    y = np.array(traindt[forecast_col])
    y = scale(y)
    print(len(X), len(y))

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X, y)

    forecast_set =  clf.predict(newX)
    traindt['Forecast'] = np.nan

    last_date = traindt.iloc[-1].name
    last_unix = last_date.timestamp()
    one_week = 86400*7
    next_unix = last_unix + one_week

    for i in forecast_set:
        next_date = datetime.fromtimestamp(next_unix)
        next_unix += 86400
        traindt.loc[next_date] = [np.nan for _ in range(len(traindt.columns)-1)]+[i]

    traindt[forecast_col].plot()
    traindt['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend(loc=1)
    plt.show()