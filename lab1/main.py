import pydot
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib

matplotlib.rc('font', family='Verdana')

data = pd.read_excel('credit.xls')
target = data.loc[:, 'kredit']
features = data.loc[:, 'laufkont':'gastarb']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=100)

clf5 = tree.DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.,
    max_features=None,
    random_state=42,
    max_leaf_nodes=None,
    min_impurity_split=1e-30,
    class_weight=None,
    presort=False
)
clf5.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf5, feature_names=list(features.columns.values),
                                class_names=['credit', 'no_credit'],
                                out_file='small_tree.dot', filled=True)
(graph,) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('gini_default.png')
print('default:', accuracy_score(y_test, clf5.predict(X_test)))
scores = cross_val_score(clf5, X_test, y_test, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()

# clf1 = tree.DecisionTreeClassifier(
#     class_weight=None, criterion='entropy', max_depth=3,
#     max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
#     min_samples_split=2, min_weight_fraction_leaf=0.0,
#     presort=False, random_state=100, splitter='best'
# )
# clf1.fit(X_train, y_train)
# dot_data = tree.export_graphviz(clf1, feature_names=list(features.columns.values),
#                                 class_names=['credit', 'no_credit'],
#                                 out_file='small_tree.dot', filled=True)
# (graph,) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('entropy.png')
# print('entropy:', accuracy_score(y_test, clf1.predict(X_test)))
# scores = cross_val_score(clf1, X_test, y_test, cv=5, scoring='f1')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print()
#
# clf2 = tree.DecisionTreeClassifier(
#     class_weight=None, criterion='gini', max_depth=3,
#     max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
#     min_samples_split=2, min_weight_fraction_leaf=0.0,
#     presort=False, random_state=100, splitter='best'
# )
# clf2.fit(X_train, y_train)
# dot_data = tree.export_graphviz(clf2, feature_names=list(features.columns.values),
#                                 class_names=['credit', 'no_credit'],
#                                 out_file='small_tree.dot', filled=True)
# (graph,) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('gini.png')
# print('gini:', accuracy_score(y_test, clf2.predict(X_test)))
# scores = cross_val_score(clf2, X_test, y_test, cv=5, scoring='f1')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print()
#
# clf3 = tree.DecisionTreeClassifier(
#     class_weight=None, criterion='gini', max_depth=3,
#     max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
#     min_samples_split=2, min_weight_fraction_leaf=0.0,
#     presort=False, random_state=100, splitter='random'
# )
# clf3.fit(X_train, y_train)
# dot_data = tree.export_graphviz(clf3, feature_names=list(features.columns.values),
#                                 class_names=['credit', 'no_credit'],
#                                 out_file='small_tree.dot', filled=True)
# (graph,) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('gini_random.png')
# print('gini with random:', accuracy_score(y_test, clf3.predict(X_test)))
# scores = cross_val_score(clf3, X_test, y_test, cv=5, scoring='f1')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print()
#
clf4 = tree.DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=60,
    max_features='sqrt'
)
from sklearn.model_selection import GridSearchCV
tree_para = {
    'criterion':['gini','entropy'],
    'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
    'min_samples_split':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
    'max_features':['sqrt','log2',None]
}
clf4 = GridSearchCV(clf4, tree_para, cv=5)
clf4.fit(X_train, y_train)
# dot_data = tree.export_graphviz(clf4, feature_names=list(features.columns.values),
#                                 class_names=['credit', 'no_credit'],
#                                 out_file='small_tree.dot', filled=True)
# (graph,) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('gini_custom.png')
print('gini custom:', accuracy_score(y_test, clf4.predict(X_test)))
scores = cross_val_score(clf4, X_test, y_test, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()
