# -*- coding: utf-8 -*-
from scipy import stats
import pandas as pd
import numpy as np


# calcula MD_1
def media_w(W):
    return np.mean(W)

# calcula MD_2
def variancia_w(W):
    return np.var(W)

# calcula MD_3
def desvio_padrao_w(W):
    return np.std(W)

# calcula MD_4
def assimetria_w(W):
    return stats.skew(W)

# calcula MD_5
def curtose_w(W):
    return stats.kurtosis(W, fisher=False)

# calcula MD_6
def perc_h1(x):
    return x[0]/sum(x)

# calcula MD_7
def perc_h2(x):
    return x[1]/sum(x)

# calcula MD_8
def perc_h3(x):
    return x[2]/sum(x)

# calcula MD_9
def perc_h4(x):
    return x[3]/sum(x)

# calcula MD_10
def perc_h5(x):
    return x[4]/sum(x)

# calcula MD_11
def perc_h6(x):
    return x[5]/sum(x)

# calcula MD_12
def perc_h7(x):
    return x[6]/sum(x)

# calcula MD_13
def perc_h8(x):
    return x[7]/sum(x)

# calcula MD_14
def perc_h9(x):
    return x[8]/sum(x)

# calcula MD_15
def perc_h10(x):
    return x[9]/sum(x)



