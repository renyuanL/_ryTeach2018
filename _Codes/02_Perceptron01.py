#
# 02_Perceptron01.py
# ry@2018.10.12
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd

from ryTeach import plot_decision_regions_binary
from ryTeach import Perceptron, AdalineGD, AdalineSGD, readIris

def main():

    X,y= readIris()

    a= Perceptron()
    a.fit(X,y)

    b= AdalineGD(eta= .0001)
    b.fit(X,y)

    c= AdalineSGD()
    c.fit(X,y)
    
    for f in ['None', 'a', 'b', 'c']:
        print('classifier=',f)
        c= eval(f)
        plot_decision_regions_binary(X, y, classifier= c, resolution= .1)#, xylim= 10)
        plt.show()
        
if __name__ == '__main__':
    
    main()
#--- the end ---