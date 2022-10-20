import os
import re
import shutil
import pandas as pd
import gdown
import config
from preprocess import ArabertPreprocessor
from datetime import datetime
from sklearn import metrics

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
        
        
def rename_logs():
    time_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    for file in os.listdir(config.LOGS_PATH):
        if not bool(re.search(r'\d', file)):
            os.rename(config.LOGS_PATH + '/' + file, config.LOGS_PATH + '/' + file[:-4] + '_' + time_str + file[-4:])     


class PredTools:
    def __init__(self, df_val, model_name, drop_out, lr ,batch_size, max_len):
        self.file_test_preds = config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_predictions' +'.tsv'
        self.file_cross_validation_preds = config.LOGS_PATH + '/' + config.DOMAIN_CROSS_VALIDATION + '_predictions' +'.tsv'
        self.file_fold_preds = config.LOGS_PATH + '/' + config.DOMAIN_CROSS_VALIDATION + '_predictions' + '_fold' +'.tsv'
        self.df_val = df_val
        self.list_df = []
        self.model_name = model_name.split("/")[-1]
        self.drop_out = drop_out
        self.lr = lr
        self.batch_size = batch_size
        self.max_len = max_len
    
    def hold_epoch_preds(self, epoch, pred_val, targ_val=1001, fold=1):
        self.test_model = True if targ_val==1001 else False
        
        # pred columns name
        pred_col = self.model_name + '_' + str(self.drop_out) + '_' + str(self.lr) + '_' + str(self.batch_size) + '_' + str(self.max_len) + '_' + str(epoch)
        pred_val = remove_from_list(pred_val)
        
        if epoch == 1:
            self.df_fold_preds = pd.DataFrame({'text':self.df_val[config.DATASET_TEXT].values,
                                'target':targ_val,
                                'fold':fold,
                                pred_col:pred_val})
        else:
            self.df_fold_preds[pred_col] = pred_val
        
    def concat_fold_preds(self):
        # concat folder's predictions
        if os.path.isfile(self.file_fold_preds):
            df_saved = pd.read_csv(self.file_fold_preds, sep='\t')
            self.df_fold_preds = pd.concat([df_saved, self.df_fold_preds], ignore_index=True)
            
        # save folder preds
        self.df_fold_preds.to_csv(self.file_fold_preds, index=False, sep='\t')
    
    def save_preds(self):
        file_path = self.file_test_preds if self.test_model else self.file_cross_validation_preds
        
        if os.path.isfile(file_path):
            self.df_preds = pd.read_csv(file_path, sep='\t')
            self.df_fold_preds = pd.merge(self.df_preds, self.df_fold_preds, on=['text','target','fold'], how='outer')
            
        # save cross_validation preds
        self.df_fold_preds.to_csv(file_path, index=False, sep='\t')
        
        # delete folder preds
        if os.path.isfile(self.file_fold_preds):
            os.remove(self.file_fold_preds)
            

def remove_from_list(pylist):
        return [item[0] for item in pylist]
            
            
            
def convert(preds, threshold=0.5):
    labels = []
    for pred in preds:
        if pred[0] > threshold:
            labels.append([1])
        else:
            labels.append([0])
    return labels