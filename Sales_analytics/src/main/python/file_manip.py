import json, os

def inputPath():
    return './src/input/'

def configPath():
    return './src/config/'

def cachedPath():
    return './cached/'

def readfile(path):
    with open(path, 'r') as jfile:
        jdata=jfile.read()
    confDict = json.loads(jdata)
    return confDict

def get_csv_config():
    dict = readfile(configPath()+'csv_config.json')
    return str(dict['to_predict']), str(dict['test_set']), str(dict['feature_set']), str(dict['train_set'])

def get_reg_config(*args,**kwargs):
    dict = readfile(configPath()+'reg_config.json')
    
    data =[]
    
    if kwargs.get('only', None) is not None: 
        for config in  kwargs.get('only', None):
            data.append(str(dict[config]))
        return tuple(data)
    return str(dict['reg_line']), str(dict['exog']), str(dict['MAX_ARIMA']), str(dict['MAX_SARIMA']), str(dict['model_saveIn']) 

def makejson(data):
    with open(cachedPath()+'AIC_logs.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)


def getAIC():
    gsresult = readfile(cachedPath()+'AIC_logs.json')
    listResAIC = []
    listResParam = []
    mergelist = []
    for res in gsresult:
        listResAIC.append(res['AIC'])
    listResAIC.sort()
    for aic in listResAIC:
        for res in gsresult:
            if aic == res['AIC']:
                listResParam.append(res['param'])

    for i in range(0, len(listResAIC)):
        mergelist.append(str(listResParam[i]) + ' -- ' + str(listResAIC[i]))
    return mergelist