import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import random_projection
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm

centers = [[1, 1], [-1, -1], [1, -1]]

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

df = pd.read_pickle('processed_data/dataframe_by_pre_process.pd')

y = labels = df["Status"].values
del df['Status']

feature_selected_by_classification = ['32166_at', '40567_at', '32598_at', '38269_at', '995_g_at',
                                      '39054_at', '34315_at', '37958_at', '1356_at', '1450_g_at',
                                      '40282_s_at', '39366_at', '41242_at', '41458_at']
df2 = df[feature_selected_by_classification]

features = df2[list(df2.columns)].values


pca = PCA(copy=True, n_components=2, whiten=False)
pca.fit(features)
X = pca.transform(features)

h = .02
classifier = svm.SVC()

svc = classifier.fit(X, labels)
classifier2 = KNeighborsClassifier(n_neighbors=1)
nearest = classifier2.fit(X, labels)


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ['SVM SVC',
          'Nearest Neighbors']

for i, clf in enumerate((svc, nearest)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('ax1')
    plt.ylabel('ax2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()