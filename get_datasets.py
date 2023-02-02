# -*- coding: utf-8 -*-
 
import argparse
import gzip
import os
import shutil
import tarfile
import tempfile
import zipfile
from glob import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import wget
import tensorflow as tf
from keras import datasets as kdatasets
from keras import applications
from scipy.io import arff
from skimage import io, transform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import metrics
from timeit import default_timer as timer 

def download_file(urls, base_dir, name):
    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

        for url in urls:
            wget.download(url, out=dir_name)


def save_dataset(name, tipo, X, y):
    n_samples = metrics.metric_dc_num_samples(X)
    n_features = metrics.metric_dc_num_features(X)
    n_classes = 0 #n_classes = metrics.metric_dc_num_classes(y)
    balanced = 0 #balanced = metrics.metric_dc_dataset_is_balanced(y)
    dim_int = 0 #metrics.metric_dc_intrinsic_dim(X)

    print(name, n_samples, n_features, n_classes, balanced, dim_int, X.shape)

    #for l in np.unique(y):
    #    print('-->', l, np.count_nonzero(y == l))

    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.astype('float32'))

    # cria metadados pra pegar MA1
    arq = open(os.path.join(dir_name, 'meta.txt'), 'w')
    arq.write('Dataset: ' + name + '\n' + 'Tipo: ' + tipo + '\n' + 'Instancias: ' + str(n_samples) + '\n' + 'Dimensões: ' + str(n_features))

    np.save(os.path.join(dir_name, 'X.npy'), X)
    np.save(os.path.join(dir_name, 'y.npy'), y)

    np.savetxt(os.path.join(dir_name, 'X.csv.gz'), X, delimiter=',')
    np.savetxt(os.path.join(dir_name, 'y.csv.gz'), y, delimiter=',')


def remove_all_datasets(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

# Banco de dados bank

def process_bank():
    bank = zipfile.ZipFile('data/bank/bank-additional.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    bank.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(
        tmp_dir.name, 'bank-additional', 'bank-additional-full.csv'), sep=';')

    y = np.array(df['y'] == 'yes').astype('uint8')
    X = np.array(pd.get_dummies(df.drop('y', axis=1)))

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('bank', 'tabular', X, y)

def process_cnae9():
    df = pd.read_csv('data/cnae9/CNAE-9.data', header=None)
    y = np.array(df[0])
    X = np.array(df.drop(0, axis=1))
    save_dataset('cnae9', 'texto', X, y)

def process_imdb():
    imdb = tarfile.open('data/imdb/aclImdb_v1.tar.gz', 'r:gz')
    tmp_dir = tempfile.TemporaryDirectory()
    imdb.extractall(tmp_dir.name)

    pos_files = glob(os.path.join(
        tmp_dir.name, 'aclImdb/train/pos') + '/*.txt')
    pos_comments = []

    neg_files = glob(os.path.join(
        tmp_dir.name, 'aclImdb/train/neg') + '/*.txt')
    neg_comments = []

    for pf in pos_files:
        with open(pf, 'r', encoding='utf-8') as f:
            pos_comments.append(' '.join(f.readlines()))

    for nf in neg_files:
        with open(nf, 'r', encoding='utf-8') as f:
            neg_comments.append(' '.join(f.readlines()))

    comments = pos_comments + neg_comments
    y = np.zeros((len(comments),)).astype('uint8')
    y[:len(pos_comments)] = 1

    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=700)
    X = tfidf.fit_transform(comments).todense()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.13, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('imdb', 'texto', X, y)


def process_sentiment():
    sent = zipfile.ZipFile('data/sentiment/sentiment labelled sentences.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    sent.extractall(tmp_dir.name)

    files = ['amazon_cells_labelled.txt',
             'imdb_labelled.txt', 'yelp_labelled.txt']
    dfs = []

    for f in files:
        dfs.append(pd.read_table(os.path.join(
            tmp_dir.name, 'sentiment labelled sentences', f), sep='\t', header=None))

    df = pd.concat(dfs)
    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=200)

    y = np.array(df[1]).astype('uint8')
    X = tfidf.fit_transform(list(df[0])).todense()
    save_dataset('sentiment', 'texto', X, y)


def process_cifar10():
    (X, y), (_, _) = tf.keras.datasets.cifar10.load_data()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.065, random_state=42, stratify=y)
    X = X_train
    y = y_train

    X = X[:,:,:,1]

    save_dataset('cifar10', 'imagem', X.reshape((-1, 32 * 32)), y.squeeze())


def process_fashionmnist():
    (X, y), (_, _) = kdatasets.fashion_mnist.load_data()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('fashion_mnist', 'imagem', X.reshape((-1, 28 * 28)), y.squeeze())

def process_zfmd():
    fmd = zipfile.ZipFile('data/fmd/FMD.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    fmd.extractall(tmp_dir.name)

    fmd_shape = (384, 512, 3)

    images = dict()

    for d in sorted(glob(os.path.join(tmp_dir.name, 'image') + '/*')):
        class_name = os.path.basename(d)

        images[class_name] = []

        for img in glob(d + '/*.jpg'):
            im = io.imread(img)
            if im.shape == fmd_shape:
                images[class_name].append(im)

    image_arrays = []
    label_arrays = []

    for i, c in enumerate(sorted(images.keys())):
        image_arrays.append(np.array(images[c]))
        labels = np.zeros((len(images[c]),)).astype('uint8')
        labels[:] = i
        label_arrays.append(labels)

    model = applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False, weights='imagenet', input_shape=fmd_shape, pooling='max')

    X = np.vstack(image_arrays)
    y = np.hstack(label_arrays)

    X = X / 255.0
    X = model.predict(X)

    save_dataset('fmd', 'imagem', X, y)


def process_svhn():
    data = sio.loadmat('data/svhn/train_32x32.mat')

    X = np.rollaxis(data['X'], 3, 0)
    X = X[:,:,:,1].reshape((-1, 32*32))
    y = data['y'].squeeze()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.01, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('svhn', 'imagem', X, y)


def process_seismic():
    data, _ = arff.loadarff('data/seismic/seismic-bumps.arff')
    df = pd.DataFrame.from_records(data)

    df['seismic'] = df['seismic'].str.decode("utf-8")
    df['seismoacoustic'] = df['seismoacoustic'].str.decode("utf-8")
    df['shift'] = df['shift'].str.decode("utf-8")
    df['ghazard'] = df['ghazard'].str.decode("utf-8")
    df['class'] = df['class'].str.decode("utf-8")

    y = np.array((df['class'] == '1').astype('uint8'))
    X = np.array(pd.get_dummies(df.drop('class', axis=1)))

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.25, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('seismic', 'tabular', X, y)

def process_epileptic():
    df = pd.read_csv('data/epileptic/data.csv', index_col=None)
    y = np.array(df['y'])
    X = np.array(df.drop(['y', 'Unnamed: 0'], axis=1))

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.5, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('epileptic', 'tabular', X, y)


def process_spambase():
    df = pd.read_csv('data/spambase/spambase.data',
                     header=None, index_col=None)
    y = np.array(df[57]).astype('uint8')
    X = np.array(df.drop(57, axis=1))
    
    save_dataset('spambase', 'texto', X, y)


def process_sms():
    sms = zipfile.ZipFile('data/sms/smsspamcollection.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    sms.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(
        tmp_dir.name, 'SMSSpamCollection'), sep='\t', header=None)
    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=500)

    y = np.array(df[0] == 'spam').astype('uint8')
    X = tfidf.fit_transform(list(df[1])).todense()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.15, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('sms', 'texto', X, y)


def process_hatespeech():
    df = pd.read_csv('data/hatespeech/labeled_data.csv')
    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=100)

    y = np.array(df['class']).astype('uint8')
    X = tfidf.fit_transform(list(df['tweet'])).todense()
    
    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.13, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('hatespeech', 'texto', X, y)


def process_secom():
    df = pd.read_csv('data/secom/secom.data', sep=' ', header=None)
    labels = pd.read_csv('data/secom/secom_labels.data', sep=' ', header=None)

    y = np.array(labels[0])
    X = np.array(df)
    X[np.isnan(X)] = 0.0
    save_dataset('secom', 'tabular', X, y)


def process_har():
    har = zipfile.ZipFile('data/har/UCI HAR Dataset.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    har.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(tmp_dir.name, 'UCI HAR Dataset', 'train',
                                  'X_train.txt'), header=None, delim_whitespace=True)
    labels = pd.read_csv(os.path.join(tmp_dir.name, 'UCI HAR Dataset', 'train',
                                      'y_train.txt'), header=None, delim_whitespace=True)

    y = np.array(labels[0]).astype('uint8')
    X = np.array(df)

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.1, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('har', 'tabular', X, y)

def process_coil20():
    coil20 = zipfile.ZipFile('data/coil20/coil-20-proc.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    coil20.extractall(tmp_dir.name)

    file_list = sorted(glob(tmp_dir.name + '/coil-20-proc/*.png'))

    img_side = 20

    X = np.zeros((len(file_list), img_side, img_side))
    y = np.zeros((len(file_list),)).astype('uint8')

    for i, file_name in enumerate(file_list):
        label = int(os.path.basename(file_name).split(
            '__')[0].replace('obj', ''))

        tmp = io.imread(file_name)
        tmp = transform.resize(tmp, (img_side, img_side), preserve_range=True)

        X[i] = tmp / 255.0
        y[i] = label

    save_dataset('coil20', 'imagem', X.reshape((-1, img_side * img_side)), y)


def process_orl():
    orl = zipfile.ZipFile('data/orl/att_faces.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    orl.extractall(tmp_dir.name)

    subjects = sorted(glob(tmp_dir.name + '/s*'))

    img_h = 112 // 5
    img_w = 92 // 5

    X = np.zeros((len(subjects * 10), img_h, img_w))
    y = np.zeros((len(subjects * 10),)).astype('uint8')

    for i, dir_name in enumerate(subjects):
        label = int(os.path.basename(dir_name).replace('s', ''))

        for j in range(10):
            tmp = io.imread(dir_name + '/%d.pgm' % (j + 1))
            tmp = transform.resize(tmp, (img_h, img_w), preserve_range=True)
            X[i] = tmp / 255.0
            y[i] = label

    save_dataset('orl', 'imagem', X.reshape((-1, img_h * img_w)), y)


def process_hiva():
    hiva = zipfile.ZipFile('data/hiva/HIVA.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    hiva.extractall(tmp_dir.name)

    X = np.loadtxt(tmp_dir.name + '/HIVA/hiva_train.data')
    y = np.loadtxt(tmp_dir.name + '/HIVA/hiva_train.labels')

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('hiva', 'tabular', X, y)


'''
def process_spatial():
    df = pd.read_csv('data/spatial/3D_spatial_network.csv', header=None)
    y = np.array(df[0])
    X = np.array(df.drop(0, axis=1))
    save_dataset('spatial', 'tabular', X, y)


def process_libra8():
    df = pd.read_csv('data/libra8/movement_libras_8.csv', header=None)
    c = df.shape[1]
    y = np.array(df[c - 1])
    X = np.array(df.drop(c-1, axis=1))
    save_dataset('libra8', 'tabular', X, y)

def process_acute_inflammations():
    df = pd.read_table('data/acute/diagnosis.data', delim_whitespace=True, header=None)
    c = df.shape[1]
    y = np.array(df.drop(columns=[0,1,2,3,4,5]))
    X = np.array(df.drop(columns=[6,7]))
    save_dataset('acute', 'tabular', X, y)
    return df

def process_water_treatement():
    df = pd.read_table("data/water-treatment/water-treatment.csv", sep=",", header=None)
    #df = df.drop(columns=[0])  
    X = np.array(df)
    y = []
    save_dataset('water-treatment', 'tabular', X, y)  
    return df


def process_zoo:
    df = pd.read_table("c:/desenvolvimento/teste/zoo.data", sep=",", header=None)# names=['A','B','C','D','E','F','G','H'])
    df = df.drop(columns=[17])   
    labelencoder_df = LabelEncoder()
    df[0] = labelencoder_df.fit_transform(df[0]) 
    return df, [], 'tabular', dataset_name

    elif dataset_name == 'agua': 
        df = pd.read_table("c:/desenvolvimento/teste/agua.csv", sep=",", header=None)# names=['A','B','C','D','E','F','G','H'])
        return df, [], 'tabular', dataset_name
    elif dataset_name == 'anneling':
        df = pd.read_table("c:/desenvolvimento/teste/anneal.data", sep=",", header=None)# names=['A','B','C','D','E','F','G','H'])
        df = df.drop(columns=[38])    
        labelencoder_df = LabelEncoder()
        #5,9,10,11,13,14,15,16,17,19,20,21,23,24,26,27,31,35
        df[0] = labelencoder_df.fit_transform(df[0])
        df[1] = labelencoder_df.fit_transform(df[1])
        df[2] = labelencoder_df.fit_transform(df[2])
        df[5] = labelencoder_df.fit_transform(df[5])
        df[6] = labelencoder_df.fit_transform(df[6])
        df[9] = labelencoder_df.fit_transform(df[9])
        df[10] = labelencoder_df.fit_transform(df[10])
        df[11] = labelencoder_df.fit_transform(df[11])
        df[13] = labelencoder_df.fit_transform(df[13])
        df[14] = labelencoder_df.fit_transform(df[14])
        df[15] = labelencoder_df.fit_transform(df[15])
        df[16] = labelencoder_df.fit_transform(df[16])
        df[17] = labelencoder_df.fit_transform(df[17])
        df[19] = labelencoder_df.fit_transform(df[19])
        df[20] = labelencoder_df.fit_transform(df[20])
        df[21] = labelencoder_df.fit_transform(df[21])
        df[23] = labelencoder_df.fit_transform(df[23])
        df[24] = labelencoder_df.fit_transform(df[24])
        df[26] = labelencoder_df.fit_transform(df[26])
        df[27] = labelencoder_df.fit_transform(df[27])
        df[31] = labelencoder_df.fit_transform(df[31])
        df[35] = labelencoder_df.fit_transform(df[35])
        return df, [], 'tabular', dataset_name
    else:
'''    

# código
if __name__ == '__main__':
    base_dir = './data'

    datasets = dict()

    '''
    datasets['spatial'] = [
        'https://nrvis.com/data/mldata/3D_spatial_network.csv']
    datasets['libra8'] = [
        'https://nrvis.com/data/mldata/movement_libras_8.csv']
    datasets['acute'] = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data']
    datasets['water-treatment'] = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/water-treatment.data']
    '''
    datasets['cnae9'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data']
    datasets['fmd'] = ['http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip']
    datasets['svhn'] = [
        'http://ufldl.stanford.edu/housenumbers/train_32x32.mat']
    datasets['bank'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip']
    datasets['seismic'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff']
    # datasets['hepmass'] = [
    #     'http://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz']
    datasets['epileptic'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv']
    # datasets['gene'] = [
    #     'https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz']
    datasets['spambase'] = ['https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
                            'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names']
    datasets['sms'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip']
    datasets['hatespeech'] = [
        'https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv?raw=true']
    # datasets['efigi'] = ['https://www.astromatic.net/download/efigi/efigi_png_gri-1.6.tgz',
    #                      'https://www.astromatic.net/download/efigi/efigi_tables-1.6.2.tgz']
    datasets['imdb'] = [
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
    datasets['sentiment'] = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment labelled sentences.zip']
    datasets['secom'] = ['http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data',
                         'http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data']
    datasets['har'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip']
    # datasets['p53'] = [
    #     'http://archive.ics.uci.edu/ml/machine-learning-databases/p53/p53_new_2012.zip']
    datasets['coil20'] = [
        'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip']
    datasets['hiva'] = [
        'http://www.agnostic.inf.ethz.ch/datasets/DataAgnos/HIVA.zip']
    datasets['orl'] = [
        'http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip']

            

    parser = argparse.ArgumentParser(
        description='Projection Survey Dataset Downloader')

    parser.add_argument('-d', action='store_true', help='delete all datasets')
    parser.add_argument('-s', action='store_true',
                        help='skip download, assume files are in place')
    args, unknown = parser.parse_known_args()

    if args.d:
        print('Removing all datasets')
        remove_all_datasets(base_dir)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    if not args.s:
        print('Downloading all datasets')
        for name, url in datasets.items():
            print('')
            print(name)
            download_file(url, base_dir, name)

    print('')
    print('Processing all datasets')

    start = timer()
    for func in sorted([f for f in dir() if f[:8] == 'process_']):
        print(str(func))
        globals()[func]()
    duration = timer() - start 
    print(duration)

