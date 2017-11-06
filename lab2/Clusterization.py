import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from matplotlib import pyplot
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split


def wgss(data, groups):
    """Within groups sum of squares (wgss)
#     Сумма квадратов расстояний от центроида до каждой точки данных
#     в многомерном пространстве.
#     Специально на английском, чтобы объяснить название функции
    """
    _data = np.array(data)
    res = 0.0
    for cluster in groups:
        inclust = _data[np.array(groups) == cluster]
        meanval = np.mean(inclust, axis=0)
        res += np.sum((inclust - meanval) ** 2)
    return res


data = pd.read_excel('credit.xls')
target = data.loc[:, 'kredit']
features = data.loc[:, 'laufkont':'gastarb']
print('Total counts:\n', target.value_counts())
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
X_train = X_train.as_matrix()
data_dist = pdist(X_train, 'euclidean')
data_linkage = linkage(data_dist, method='average')

elbow = [np.nan, wgss(X_train, [1] * len(X_train[:, 1]))]
for k in range(2, 10):
    groups = fcluster(data_linkage, k, criterion='maxclust')
    elbow.append(wgss(X_train, groups))

fig = figure()
ax = fig.add_subplot('121')  # 2 графика в строке, выбираем первый график
elbow = np.array(elbow)  # Пусть будет numpy массив, удобней...
ax.plot(elbow / np.nanmax(elbow), 'o', ls='solid')
ax.set_xlim([0, 10])
ax.set_ylim([0, 1.2])
ax.set_title('Сумма внутригрупповых вариаций#1')
ax.set_xlabel('Число кластеров')

ax1 = fig.add_subplot('122')  # выбираем второй график в строке

ax1.plot((elbow[1] - elbow) / np.nanmax(elbow), 'o', ls='solid')
ax1.set_xlim([0, 10])
ax1.set_ylim([0, 1.2])
ax1.set_xlabel('Число кластеров')
ax1.set_title('Доля объясняемой вариации')
pyplot.show()
