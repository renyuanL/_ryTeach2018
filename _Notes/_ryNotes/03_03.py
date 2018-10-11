>>> from sklearn import datasets
>>> import numpy as np
>>> iris = datasets.load_iris()
>>> X = iris.data[:, [2, 3]]
>>> y = iris.target


>>> nTrain= int(len(X)*0.7)
>>> nTest=  len(X) - nTrain

>>> X_train, X_test= X[0:nTrain], X[nTrain:]
>>> y_train, y_test= y[0:nTrain], y[nTrain:]

>>> len(X_train), len(X_test)

### (105, 45)

>>> from sklearn.cross_validation import train_test_split

>>> X_train, X_test, y_train, y_test= train_test_split(
...           X, y, test_size= 0.3, random_state= 0)


>>> from sklearn.preprocessing import StandardScaler
>>> sc= StandardScaler()
>>> sc.fit(X_train)
>>> X_train_std= sc.transform(X_train)
>>> X_test_std=  sc.transform(X_test)

>>> X_train.mean(axis=0), X_train.std(axis=0)
### array([ 2.99619048,  0.84761905]), 
### array([ 1.53082137,  0.61938452])

>>> X_train_std.mean(axis=0), X_train_std.std(axis=0)

### array([ -5.07530526e-17,   4.22942105e-17]), 
### array([ 1.,  1.])


>>> import matplotlib.pyplot as plt

>>> plt.scatter(X_train[:,0], X_train[:,1])
>>> plt.show()

>>> plt.scatter(X_train_std[:,0], X_train_std[:,1])
>>> plt.show()

#-----

>>> X_train_std= (X_train-X_train.mean(axis=0))/X_train.std(axis=0)
>>> X_train_std.mean(axis=0), X_train_std.std(axis=0)
### array([ -5.07530526e-17,   4.22942105e-17]), 
### array([ 1.,  1.])

#-----
>>> X_train.shape,  
... X_train.mean(axis=0).shape, 
... X_train.std(axis=0).shape

### (105, 2), 
### (2,), 
### (2,)



#-------------------------
# ndarray
#-------------------------

[[[[00,01,02,03],
   [04,05,06,07],
   [08,09,10,11]],
  [[12,13,14,15],
   [16,17,18,19],
   [20,21,22,23]]]]

array([[[[  0,   1,   2,   3,   4],
         [  5,   6,   7,   8,   9],
         [ 10,  11,  12,  13,  14],
         [ 15,  16,  17,  18,  19]],

        [[ 20,  21,  22,  23,  24],
         [ 25,  26,  27,  28,  29],
         [ 30,  31,  32,  33,  34],
         [ 35,  36,  37,  38,  39]],

        [[ 40,  41,  42,  43,  44],
         [ 45,  46,  47,  48,  49],
         [ 50,  51,  52,  53,  54],
         [ 55,  56,  57,  58,  59]]],


       [[[ 60,  61,  62,  63,  64],
         [ 65,  66,  67,  68,  69],
         [ 70,  71,  72,  73,  74],
         [ 75,  76,  77,  78,  79]],

        [[ 80,  81,  82,  83,  84],
         [ 85,  86,  87,  88,  89],
         [ 90,  91,  92,  93,  94],
         [ 95,  96,  97,  98,  99]],

        [[100, 101, 102, 103, 104],
         [105, 106, 107, 108, 109],
         [110, 111, 112, 113, 114],
         [115, 116, 117, 118, 119]]]])


array([[[[  0,   1,   2,   3,   4],     [[ 20,  21,  22,  23,  24],     [[ 40,  41,  42,  43,  44],
         [  5,   6,   7,   8,   9],      [ 25,  26,  27,  28,  29],      [ 45,  46,  47,  48,  49],
         [ 10,  11,  12,  13,  14],      [ 30,  31,  32,  33,  34],      [ 50,  51,  52,  53,  54],
         [ 15,  16,  17,  18,  19]],     [ 35,  36,  37,  38,  39]],     [ 55,  56,  57,  58,  59]]],

       [[[ 60,  61,  62,  63,  64],     [[ 80,  81,  82,  83,  84],      [[100, 101, 102, 103, 104],
         [ 65,  66,  67,  68,  69],      [ 85,  86,  87,  88,  89],       [105, 106, 107, 108, 109],
         [ 70,  71,  72,  73,  74],      [ 90,  91,  92,  93,  94],       [110, 111, 112, 113, 114],
         [ 75,  76,  77,  78,  79]],     [ 95,  96,  97,  98,  99]],      [115, 116, 117, 118, 119]]]])

        
        
        
        

        
        
        
        



