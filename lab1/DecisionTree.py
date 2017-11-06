import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydot
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_excel('credit.xls')
target = data.loc[:, 'kredit']
features = data.loc[:, 'laufkont':'gastarb']
print('Total counts:\n', target.value_counts())
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier(
    # random_state=17,
    # max_depth=12,
    # min_samples_leaf=20,
    # min_impurity_split=0.3

    class_weight=None, criterion='gini', max_depth=3,
    max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
    min_samples_split=2, min_weight_fraction_leaf=0.0,
    presort=False, random_state=42, splitter='best'
)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_test, y_test, cv=5, scoring='f1')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
l = [clf.predict(key.reshape(1, -1))[0] for key in X_test.values]
# print(confusion_matrix(y_test, l))
dot_data = tree.export_graphviz(clf, feature_names=list(features.columns.values),
                                class_names=['credit', 'no_credit'],
                                out_file='small_tree.dot', filled=True)
(graph,) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('example1.png')
# print(clf.feature_importances_)
print(accuracy_score(y_test, clf.predict(X_test)))
