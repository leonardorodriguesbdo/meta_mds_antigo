#import argparse
#import gzip
#from glob import glob
import numpy as np
import pandas as pd
#import scipy.io as sio
#from keras import datasets as kdatasets
#from keras import applications
#from scipy.io import arff
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import metrics

df = pd.DataFrame([[41,'leo',1981,'no'],[21,'chico',2021,'yes'],[110,'mutcha',1985,'no'],[100,'serena',1995,'no']], columns=["idade", "nome", "ano","y"])
print(df)
print('')

y = np.array(df['y'] == 'yes').astype('uint8')
X = np.array(pd.get_dummies(df.drop('y', axis=1)))

print(X)
print('')
print(y)
print('---train---')
n_samples = metrics.metric_dc_num_samples(X)
n_features = metrics.metric_dc_num_features(X)
n_classes = metrics.metric_dc_num_classes(y)
balanced = metrics.metric_dc_dataset_is_balanced(y)
outilei = metrics.metric_perc_outllier(X)

print(n_samples, n_features, n_classes, balanced, outilei, X.shape)
print('---escalar---')
scaler = MinMaxScaler()
X = scaler.fit_transform(X.astype('float32'))
print(X)
