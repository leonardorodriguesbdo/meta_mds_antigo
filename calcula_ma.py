# -*- coding: utf-8 -*-
import pandas as pd
import math
import numpy as np
#from scipy.stats import entropy
from sklearn.decomposition import PCA

# calcula MA_1
def tipo_dado(t):
    return t

def total_objetos(Ds):
    return Ds.size

# calcula MA_2 (samples)
def total_linhas(Ds):
    return Ds.shape[0]

# calcula MA_3 (Features)
def total_dimensoes(Ds):
    # junta X com y formando um unico dataframe
    #Ds = pd.DataFrame(X)
    #Ds[X.shape[1]] = y

    return Ds.shape[1]

def metric_dc_num_classes(y):
    return len(np.unique(y))

# calcula MA_4
def taxa_dim_intrinseca(Ds):
    pca = PCA()
    pca.fit(Ds)

    #return np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1
    return (np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1) / Ds.shape[1]    

# calcula MA_5
def razao_dispersao(Ds):
    return 1.0 - (np.count_nonzero(Ds) / float(Ds.shape[0] * Ds.shape[1]))

# calcula MA_6
def porc_outiliers(X, y):
    # junta X com y formando um unico dataframe
    Ds = pd.DataFrame(X)
    #Ds[X.shape[1]] = y

    # Calculating Outliers based on IQR method
    Q1 = Ds.quantile(0.25)
    Q3 = Ds.quantile(0.75)

    IQR = Q3 - Q1
    dataOutliers = (Ds < (Q1 - 1.5 * IQR)) | (Ds > (Q3 + 1.5 * IQR))
    outliers = 0

    for col in range(0, dataOutliers.shape[1]):
        out_col = dataOutliers[col].value_counts()
        for x in range(0, out_col.shape[0]):
            #print(out_col.index[x])
            if out_col.index[x]:
                outliers += out_col[x]
    
    return outliers

# calcula MA_7
def corr_media_atrib_continuos(DsC):
    #Ds = pd.DataFrame(X)
    #Ds[X.shape[1]] = y
    #dadosContinuo = Ds
    
    correlacao = np.absolute(DsC.corr(method = 'pearson'))
    if correlacao.empty :
        correlacao = 0
    correl = np.mean(correlacao)
    return np.mean(correl)

# calcula MA_8
def assim_media_atrib_continuos(DsC):
    #Ds = pd.DataFrame(X)
    #Ds[X.shape[1]] = y
    #dadosContinuo = Ds

    assimetria = DsC.skew(axis = 0, skipna = True)
    if assimetria.empty :
        assimetria = 0
    return np.mean(assimetria)

# calcula MA_9
def curt_media_atrib_continuos(DsC):
    #Ds = pd.DataFrame(X)
    #Ds[X.shape[1]] = y
    #dadosContinuo = dados_continuos(Ds)

    curt = DsC.kurtosis()
    if curt.empty :
        curt = 0
    return np.mean(curt)

def dados_continuos(Ds):
    Ds = pd.DataFrame(Ds)
    cf = 0
    cc = 0
    discrete = 0  #Nem precisa *
    total_obj = total_objetos(Ds)
    dataContinuos = Ds.copy()

    for i in range(0,Ds.shape[1]):
        perc = Ds[i].value_counts(normalize=True, dropna= False)*100
        
        # Remoção de atributos constantes
        if perc.values[0] == 100:
            Ds = Ds.drop([i],axis = 1)
            dataContinuos = dataContinuos.drop([i],axis = 1)
            #dataDiscrets = dataDiscrets.drop([i],axis = 1)
            #print('Encontrado atributos constantes')

        # Remoção de atributos identificadores
        elif perc.values[0] == (1/Ds.shape[0])*100:
            Ds = Ds.drop([i],axis = 1)
            dataContinuos = dataContinuos.drop([i],axis = 1)
            #dataDiscrets = dataDiscrets.drop([i],axis = 1)
            #print('Encontrado atributo identificador')

        # Remoção de atributos faltantes
        elif (Ds[i].isna().sum()/total_obj)*100 >= 40:
            Ds = Ds.drop([i],axis = 1)
            dataContinuos = dataContinuos.drop([i],axis = 1)
            #dataDiscrets = dataDiscrets.drop([i],axis = 1)
            #print('Encontrado mais de 40% de atributos faltantes')
        
        # Diferencia atributos continuos de discretos de acordo regra do artigo ferrari
        else:
            if Ds[i].dtype == 'float' or Ds[i].dtype == 'float64':
                #print('continuo')
                #cc += 1
                cf += 1
                #dataDiscrets = dataDiscrets.drop([i],axis = 1)
            elif Ds[i].dtype == 'int64' or Ds[i].dtype == 'int':
                #ci += 1
                if (Ds[i].nunique()/total_obj)*100 < 30:
                    #discrete += 1
                    #cd += 1
                    # calculate the entropy
                    #entr.append(pandas_entropy(perc))
                    # calculate the concetration
                    #concetr.append(concentration(dataP, i))
                    dataContinuos = dataContinuos.drop([i],axis = 1)
                else:
                    #continuos                
                    #dataDiscrets = dataDiscrets.drop([i],axis = 1)
                    cc += 1
    
    return dataContinuos
