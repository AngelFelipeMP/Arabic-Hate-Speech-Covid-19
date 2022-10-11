import os
import shutil
import pandas as pd
import gdown
from preprocess import ArabertPreprocessor
# import config

def download_data(data_path, data_urls):
    # create a data folder
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    
    for url in data_urls:
        #download data folders to current directory
        gdown.download_folder(url, quiet=True)
        sorce_folder = os.getcwd() + '/' + 'data'
        
        # move datasets to the data folder
        file_names = os.listdir(sorce_folder)
        for file_name in file_names:
            shutil.move(os.path.join(sorce_folder, file_name), data_path)
            
        # delete data folders from current directory
        shutil.rmtree(sorce_folder)
        

def process_cerist2022_data(data_path, text_col, label_col):
    arabic_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv2", keep_emojis = True)
    files = [f for f in os.listdir(data_path) if 'processed' not in f]
    
    for file in files:
        df = pd.read_csv(data_path + '/' + file, sep='\t', index_col=0)
        print(df.head())
        
        # preprocess text
        df[text_col] = df.loc[:, text_col].apply(lambda x: arabic_prep.preprocess(x))
        print(df.head())
        
        # label text -> num
        df.replace({label_col:{'Not Hatefull':0, 'Hatefull':1}}, inplace=True)
        print(df.head())
        

        df.to_csv(data_path + '/' + file[:-4] + '_preprocessed' + '.tsv', sep='\t',  index_label=0)