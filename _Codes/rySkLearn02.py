#
# rySkLearn02.py
# ry@2018.10.19
#

import numpy                as np
import matplotlib.pyplot    as plt
from   matplotlib.colors    import ListedColormap

from sklearn                import datasets
from sklearn.linear_model   import Perceptron

from ryTeach import plot_decision_regions

#------------------------------------------

def plot_with_label_and_show():
    plt.xlabel('petalLength [standardized]')
    plt.ylabel('petalWidth  [standardized]')
    plt.legend()
    plt.show()

# 讀進資料
iris= datasets.load_iris()
X=    iris.data[:, [0, 2]]
y=    iris.target

### Training a logistic regression model 

from sklearn.linear_model import LogisticRegression

lr= LogisticRegression(C= 1.0, random_state= 1)
lr.fit(X, y)

plot_decision_regions(X, y, classifier= lr)

plot_with_label_and_show()


### Training a SVM (linear kernel) 

from sklearn.svm import SVC

svm= SVC(kernel= 'linear', 
         C= 1.0, 
         random_state= 1)
svm.fit(X, y)

plot_decision_regions(X, y, classifier= svm)

plot_with_label_and_show()



### Training a SVM (rbf kernel)

svm= SVC(kernel= 'rbf', 
         C= 1.0, 
         random_state= 1, 
         gamma= 1.0)
svm.fit(X, y)

plot_decision_regions(X, y, classifier= svm)

plot_with_label_and_show()