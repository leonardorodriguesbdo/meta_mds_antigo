# -*- coding: utf-8 -*-i
import os
import csv
import time
import timeit
import numpy as np
import pandas as pd
import scipy as sci
from scipy import stats, spatial
import calcula_ma as cma
import calcula_md as cmd
from metrics import *
import vp
import math
from sklearn import datasets
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sys import stdout
import projections
import joblib


def lista_projections():
    return sorted(projections.all_projections.keys())
    
def run_eval(dataset_name, projection_name, X, y, output_dir):
    # TODO: add noise adder
    global DISTANCES

    dc_results = dict()
    pq_results = dict()
    projected_data = dict()

    dc_results['original'] = eval_dc_metrics(
       X=X, y=y, dataset_name=dataset_name, output_dir=output_dir)
    
    proj_tuple = projections.all_projections[projection_name]
    proj = proj_tuple[0]
    grid_params = proj_tuple[1]
    
    grid = ParameterGrid(grid_params)
    
    for params in grid:
        id_run = proj.__class__.__name__ + '-' + str(params)
        proj.set_params(**params)

        print('-----------------------------------------------------------------------')
        print(projection_name, id_run)

        X_new, y_new, result = projections.run_projection(
            proj, X, y, id_run, dataset_name, output_dir)
        
        print(result)
        pq_results[id_run] = result
        projected_data[id_run] = dict()
        projected_data[id_run]['X'] = X_new
        projected_data[id_run]['y'] = y_new

    results_to_dataframe(dc_results, dataset_name).to_csv(
        '%s/%s_%s_dc_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    results_to_dataframe(pq_results, dataset_name).to_csv(
        '%s/%s_%s_pq_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    joblib.dump(projected_data, '%s/%s_%s_projected.pkl' %
                (output_dir, dataset_name, projection_name))
    joblib.dump(DISTANCES, '%s/%s_%s_distance_files.pkl' %
                (output_dir, dataset_name, projection_name))
    



def carrega_dataset(dataset_name):
        data_dir = os.path.join('data', dataset_name)

        #Pegar o meta-atributo MA1
        arq = open(os.path.join(data_dir, 'meta.txt'), 'r')
        for linha in arq:
            if linha[:6] == 'Tipo: ':
                tipo = linha[6:]

        #print([l for l in arq if l[:6] == 'Tipo: '])        
        
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))

        return X, y, tipo.replace('\n', ''), dataset_name

# Normalização do dataset. Escala [0,1]
def dataset_Normalizado(Ds):
    df_min_max_scaled = Ds.copy() 
    for column in df_min_max_scaled.columns: 
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())     
    return df_min_max_scaled

# Vetor d da similaridade entre as instancias
def vetor_dist_normalizado(d):
    df_min_max_scaled = d.copy() 
    df_min_max_scaled = (df_min_max_scaled - df_min_max_scaled.min()) / (df_min_max_scaled.max() - df_min_max_scaled.min())     
    return df_min_max_scaled

def meta_atributos(nome, tipo, DsN, yN, wl, projection_name):
    #X, y = load_dataset('libra8')
        arredonda = 4

        #Nome base
        ma = d

        # MA_1 tipo de dado
        ma1 = tipo #cma.tipo_dado(t)
        
        # MA_2 numero de linhas
        #ma2 = cma.total_linhas(X) 
        ma2 = round(cma.total_linhas(DsN), arredonda) #round(math.log(cma.total_linhas(Ds), 2), arredonda)

        # MA_3 numero de dimensões
        #ma3 = cma.total_dimensoes(X, y)
        ma3 = round(cma.total_dimensoes(DsN), arredonda)  #X, y), 2), arredonda) #round(math.log(cma.total_dimensoes(Ds), 2), arredonda)  #X, y), 2), arredonda)

        # MA_4 taxa dimensionalidade intrinseca
        ma4 = round(cma.taxa_dim_intrinseca(DsN), arredonda)
        
        # MA_5 razão de disoersão
        ma5 = round(cma.razao_dispersao(DsN), arredonda)

        # MA_6 porcentagem de outiliers
        ma6 = round(cma.porc_outiliers(DsN, yN)/ cma.total_linhas(X), arredonda)

        # Retorna dataset apenas dos atributos continuos
        DsContinuo = cma.dados_continuos(DsN)
        # MA_7 correlação media absluta entre atributos continuos
        ma7 = round(cma.corr_media_atrib_continuos(DsContinuo), arredonda)

        # MA_8 Assimetria média de atributos continuos
        ma8 = round(cma.assim_media_atrib_continuos(DsContinuo), arredonda)
               
        # MA_9 Curtose media dos atributos continuos
        ma9 = round(cma.curt_media_atrib_continuos(DsContinuo), arredonda)

        #dados.append([ma1, ma2, ma3, ma4, ma5, ma6, ma7, ma8, ma9])

        # MD_1 Media de w'
        md1 = round(cmd.media_w(wl), arredonda)
        # MD_2 Variancia de w'
        md2 = round(cmd.variancia_w(wl), arredonda)
        # MD_3 Desvio padrão de w'
        md3 = round(cmd.desvio_padrao_w(wl), arredonda)
        # MD_4 Assimetria de w'
        md4 = round(cmd.assimetria_w(wl), arredonda)
        # MD_5 Curtose de w'
        md5 = round(cmd.curtose_w(wl), arredonda)
        
        d_hist = np.histogram(wl)#,bins=10,range=(0.0,1.0))
        x = d_hist[0]

        # MD_6 percentual intervalo [0,0.1]'
        md6 = round(cmd.perc_h1(x), arredonda)
        # MD_7 percentual intervalo (0.1,0.2]'
        md7 = round(cmd.perc_h2(x), arredonda)
        # MD_8 percentual intervalo (0.2,0.3]'
        md8 = round(cmd.perc_h3(x), arredonda)
        # MD_9 percentual intervalo (0.3,0.4]'
        md9 = round(cmd.perc_h4(x), arredonda)
        # MD_10 percentual intervalo (0.4,0.5]'
        md10 = round(cmd.perc_h5(x), arredonda)
        # MD_11 percentual intervalo (0.5,0.6]'
        md11 = round(cmd.perc_h6(x), arredonda)
        # MD_12 percentual intervalo (0.6,0.7]'
        md12 = round(cmd.perc_h7(x), arredonda)
        # MD_13 percentual intervalo (0.7,0.8]'
        md13 = round(cmd.perc_h8(x), arredonda)
        # MD_14 percentual intervalo (0.8,0.9]'
        md14 = round(cmd.perc_h9(x), arredonda)
        # MD_15 percentual intervalo (0.9,1]'
        md15 = round(cmd.perc_h10(x), arredonda)

        #Z-Score = (d[0]-md1) / md3
        z = stats.zscore(wl)
        y1 = sum(0 <= x < 1 for x in np.absolute(z))
        y2 = sum(1 <= x < 2 for x in np.absolute(z))
        y3 = sum(2 <= x < 3 for x in np.absolute(z))
        y4 = sum(3 <= x for x in np.absolute(z))
        y = y1 + y2 + y3 + y4

        # MD_16 percentual escore-z absoluto intervalo [0,1)'
        md16 = round((y1/y), arredonda)
        # MD_17 percentual escore-z absoluto intervalo [1,2)'
        md17 = round((y2/y), arredonda)
        # MD_18 percentual escore-z absoluto intervalo [2,3)'
        md18 = round((y3/y), arredonda)
        # MD_19 percentual escore-z absoluto intervalo [3,--)'
        md19 = round((y4/y), arredonda)
        
        # Extrai métricas de qualidade para cada projeção no dataset
        for p in projection_name:
            run_eval(d, p, DsN, yN, output_dir)

        dados.append([ma,ma1, ma2, ma3, ma4, ma5, ma6, ma7, ma8, ma9, md1, md2, md3, md4, md5, md6, md7, md8, md9, md10, md11, md12, md13, md14, md15, md16, md17, md18, md19])

 

if __name__ == '__main__':
    path = os.getcwd() + '\data'
    projection_name = lista_projections()
    output_dir = os.path.join('metricas_quali')
    datasets = os.listdir(path)
    dados = []
    #datasets = ['bank']
    #datasets = ['bank','cnae9','acute','water_treatement'] #, 'spatial', 'libra8','acute','water_treatement'] 
    #datasets = ['libra8', 'acute', 'agua', 'zoo'] 

    # Abre o arquivo da base de conhecimento guarda o conteúdo ja processado
    with open('base_conhecimento.csv', encoding='utf-8') as base_conhecimento:
        tabela = csv.reader(base_conhecimento, delimiter=',')
        for l in tabela:
            dados.append(l)
    
    print("Processando meta-atributos...")
    for d in datasets:
        # Verifica se o dataset ja foi processado, caso sim pula para o próximo
        if 'ok' in d:
            continue
        
        # Inicia o timer do processamento
        time_proc_bd = timeit.Timer("meta_atributos", "from __main__ import meta_atributos")
        
        X, y, tipo, nome = carrega_dataset(d)

        stdout.write(nome)
        stdout.write(' .....')
        print('.')

        DsN = X #pd.DataFrame(X)
        yN = y #pd.DataFrame(y) 
        #Ds[X.shape[1]] = y
        #DsN = Ds.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

        #DsN = Ds #dataset_Normalizado(Ds)
        #print(Ds)
        #print(DsN)

        # "vetor w' normalizado, gerado a partir de w"
        w = spatial.distance.pdist(DsN, metric='euclidean')
        wl = vetor_dist_normalizado(w)
        #print(wl)
       
        meta_atributos(nome, tipo, DsN, yN, wl, projection_name)

        #renomear a pasta sinalizando seu processamento
        nome_antigo = path + '\\' + nome
        nome_novo = path + '\\' + nome + '_ok'
        os.rename(nome_antigo, nome_novo)

        time_proc_bd = time_proc_bd.repeat(repeat = 3, number = 10000)
        print('........... {}'.format(min(time_proc_bd)), ' segundos')

    #print(dados)

#df = pd.DataFrame(dados, columns=['BD','MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6', 'MA7', 'MA8', 'MA9', 'MD1', 'MD2', 'MD3', 'MD4', 'MD5', 'MD6', 'MD7', 'MD8', 'MD9', 'MD10', 'MD11', 'MD12', 'MD13', 'MD14', 'MD15', 'MD16', 'MD17', 'MD18', 'MD19'])

# Atualiza o arquivo csv com os dados processados
f = open('base_conhecimento.csv', 'w', newline='', encoding='utf-8')
w = csv.writer(f)
for l in dados:
    w.writerow(l)
f.close()

print('--------------------------------------------------------------------')
print("Processamento Finalizado.")
#print(df)
print('--------------------------------------------------------------------')

# Convertendo dataframe para um arquivo CSV
#df.to_csv('base_conhecimento', encoding='utf-8')
#df.to_csv('base_conhecimento', encoding='utf-8', index=False)
#df.to_csv('base_conhecimento', encoding='utf-8',header=False)
#time_proc = time_proc.repeat(repeat = 3, number = 10000)
#print('Tempo total {}'.format(min(time_proc_bd)), ' segundos')


