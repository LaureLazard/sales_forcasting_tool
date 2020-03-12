import warnings
import json
import file_manip
#Data Reading and Manipulation
import pandas as pd 
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import date

#Graphical display
from PyQt5 import QtGui
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import style
from pylab import rcParams
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
from statsmodels.api import tsa
import dfgui #From https://github.com/bluenote10/PandasDataFrameGUI
def showDf(df, title):
    dfgui.show(df, title)

#Data Analysis
from sklearn import decomposition
from scipy import sparse
import scipy.stats as ss
import math
from math import sqrt
import itertools
import bisect

#Machine learning tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import sparse
def SARIMAX(dataframe, X ,p,d,q, P,D,Q, S, stationarity, invertibility):
   return tsa.statespace.SARIMAX(dataframe,exog=X, order=(p,d,q), seasonal_order=(P,D,Q, S), enforce_stationarity=stationarity,
                                    enforce_invertibility=invertibility)
from statsmodels.tsa.arima_model import ARIMAResults


def LinReg(vX, vY):
    regressor = LinearRegression()  
    regressor.fit(vX, vY)
    return regressor


#debugging and file handling
import gc
import time
import warnings
import os
import subprocess
import shutil



## Keras for deep learning
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras import regularizers
from keras import optimizers

## Performance measures
from sklearn.metrics import mean_squared_error
