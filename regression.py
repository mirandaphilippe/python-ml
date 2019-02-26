
import pandas as pd 
import quandl as qd
import math, numpy
from sklearn import preprocessing, cross_validation, svm

# Philippe Miranda
# The tutorial that this code is based in is from @sentdex Youtube Channel's.
# In the original code was not Object Oriented so i made some adaptations.
# The objective is to create an machine learning algorithm that evaluate to stock prices and try to predict
# how much the share will value on the next day.

class Regression:
    
    df = None
    forecast_column = None
    forecast_out = None

    def __init__(self):
        self.forecast_column = 'Adj. Close'
        self.setFeatures()
    
    def setFeatures(self):
        df = qd.get('WIKI/GOOGL')
        df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
        df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0 #Volatility
        df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #Daily Percent
        df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
        df.fillna(-99999, inplace=True) # If a column has no data, for the algorithm is better set a negative value rather than ignore the column
        
        self.forecast_out = int(math.ceil(0.01*len(df))) #
        df['label'] = df[self.forecast_column].shift(-self.forecast_out)
        df.dropna(inplace=True)
        self.df = df
   
    def getAllColumns(self):
        self.df = qd.get('WIKI/GOOGL')


Regression()