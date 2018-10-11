#
# rySkLearn01.py
# ry@2018.10.10
#

from sklearn import datasets
from sklearn.linear_model import Perceptron

# 讀進資料
iris= datasets.load_iris()
X= iris.data[:, [0, 2]]
y= iris.target

# 生成分類器
ppn= Perceptron(n_iter= 20, eta0= .01, random_state= 1)

# 訓練 (fit)
ppn.fit(X, y)

#plot_decision_regions(X, y, classifier= ppn)

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

# 改善分類器，努力提升正確率，
#令輸入特徵值標準化 (mean=0, std=1)
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
sc.fit(X)
X_std= sc.transform(X)

ppn.fit(X_std, y)

#plot_decision_regions(X_std, y, classifier= ppn)

theScore= ppn.score(X_std,y)
print('theScore= ', theScore)
