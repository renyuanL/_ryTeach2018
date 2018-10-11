#
# rySkLearn01.py
# ry@2018.10.10
#

import numpy                as np
import matplotlib.pyplot    as plt
from   matplotlib.colors    import ListedColormap

from sklearn                import datasets
from sklearn.linear_model   import Perceptron

from ryTeach import plot_decision_regions

# 讀進資料
iris= datasets.load_iris()
X= iris.data[:, [0, 2]]
y= iris.target

# 產生一個 Perceptron 分類器
ppn= Perceptron(n_iter= 20, eta0= .01, random_state= 1)

# 訓練 (fit)
ppn.fit(X, y)

# 畫決策區域圖
plot_decision_regions(X, y, classifier= ppn)
plt.show()

# 預測 (predict)
yHat= ppn.predict(X)

# 評估正確率 (之1)
theTrueNumber=  (y==yHat).sum()
theTotalNumber= len((y==yHat))
theAccuracy=    theTrueNumber/theTotalNumber
print('theAccuracy= ', theAccuracy)

# 評估正確率 (之2)
theScore= ppn.score(X,y)
print('theScore= ', theScore)

# 改善分類器，努力提升正確率....

#令輸入特徵值標準化 (mean=0, std=1)
import sklearn.preprocessing

sc= sklearn.preprocessing.StandardScaler()
sc.fit(X)
X_std= sc.transform(X)

# 重新訓練 (fit)
ppn.fit(X_std, y)

# 重新畫決策區域圖
plot_decision_regions(X_std, y, classifier= ppn)
plt.show()

# 重新評估正確率 (之3)
theScore= ppn.score(X_std,y)
print('theScore= ', theScore)

#-----------------------------------------------

