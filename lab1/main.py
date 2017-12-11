import pydot
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report

matplotlib.rc('font', family='Verdana')

data = pd.read_excel('credit.xls')
target = data.loc[:, 'kredit']
features = data.loc[:, 'laufkont':'gastarb']
features = SelectKBest(chi2 , k=14).fit_transform(features, target)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.7, random_state=100)
sample_weight_last_ten = abs(pd.np.random.randn(len(X_train)))
sample_weight_last_ten[5:] *= 5
sample_weight_last_ten[9] *= 15
clf4 = tree.DecisionTreeClassifier(class_weight=sample_weight_last_ten)
tree_para = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15],
    'min_samples_split': [4, 10, 15, 20, 30, 40, 50, 90, 120, 150],
    'splitter': ['best', 'random'],
    'class_weight': ['balanced', None]
}
clf4 = GridSearchCV(clf4, tree_para, cv=5)
clf4.fit(X_train, y_train)
print(clf4.best_params_)
y_pred = clf4.predict(X_test)
print(classification_report(y_test, y_pred))
# dot_data = tree.export_graphviz(clf4, feature_names=list(features.columns.values),
#                                 class_names=['credit', 'no_credit'],
#                                 out_file='small_tree.dot', filled=True)
# (graph,) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('gini_custom.png')
print('gini custom:', accuracy_score(y_test, clf4.predict(X_test)))
scores = cross_val_score(clf4, X_test, y_test, cv=2, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()
