import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
import pydot
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold

data = pd.read_excel('credit.xls')
target = data.loc[:, 'kredit']
features = data.loc[:, 'laufkont':'gastarb']
print(features.shape)