import os
#Hiper-parameters
SPLITS = 2 #5
EPOCHS = 5 #5
MAX_LEN = [32] #128
DROPOUT = [0.3]
LR = [1e-5]
BATCH_SIZE = [8] #32
TRANSFORMERS = ['aubmindlab/araelectra-base-discriminator',
                'aubmindlab/bert-base-arabertv02-twitter',
                'xlm-roberta-base',
                'bert-base-multilingual-cased', 
                'aubmindlab/aragpt2-base',
                'asafaya/albert-large-arabic']
DEVICE = 'cpu'

N_ROWS= 32 #None
SEED = 17
CODE_PATH = os.getcwd()
REPO_PATH = '/'.join(CODE_PATH.split('/')[0:-1])
DATA_PATH = REPO_PATH + '/' + 'data'
LOGS_PATH = REPO_PATH + '/' + 'logs'

DATA_URL = ['???']

DATASET_TEXT = 'Text'
DATASET_LABEL = 'Contains_Fake_Information'

DATASET_TRAIN = 'train_hate_speech_preprocessed.tsv'
DATASET_TEST = 'test_hate_speech_preprocessed.tsv'
DOMAIN_CROSS_VALIDATION = 'cross_validation'
DOMAIN_TEST = 'test'

TRAIN_WORKERS = 1
VAL_WORKERS = 1
TEST_WORKERS = 1 
LOGS_PATH = REPO_PATH + '/' + 'logs'
