import pandas as pd
import os
import config

if __name__ == "__main__":
    # get data path
    path = os.getcwd()
    repo_path = '/'.join(path.split('/')[0:-1])
    data_path = path + '/' + 'data'

    # data files
    train_file = 'train_hate_speech.tsv'
    train_file = 'train_hate_speech.tsv'

    # load data
    df_train = pd.read_csv(config.DATA_PATH + '/' + train_file, sep='\t', index_col=0)
    print(df_train.head())

    # dataset size
    print(df_train.shape)

    # label distribution
    label_distribution = df_train.groupby('Contains_Fake_Information').count()
    print(label_distribution)