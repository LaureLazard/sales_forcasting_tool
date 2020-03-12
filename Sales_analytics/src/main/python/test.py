import arima, datamanip
from time import process_time,sleep

_df = [[1,1],[12,20],[27,40]]
_exog = [[None,'no'], ['Temperature','yes']]
pred='Weekly_Sales'
train, test = datamanip.manip_data(1,1)
for exog in _exog:
    for df in _df:
        train, test = datamanip.manip_data(1,1)
        perf=[]
        for i in range(0,3):
            perf.append(arima.gridSearch(train, exog=exog[0], focused=pred))
            sleep(5)
        print ('df: %d_%d' % (df[0], df[1]))
        print ('Max_SARIMA: %d' % (3))
        print ('exog: %s' % (exog[1]))
        print ('time: {}'.format(perf))
        input("Press Enter to continue...")