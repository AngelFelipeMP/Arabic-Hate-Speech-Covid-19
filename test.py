import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
from utils import PredTools, rename_logs, convert

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore') 
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


def run(df_train, df_test, max_len, transformer, batch_size, drop_out, lr, df_results):
    
    train_dataset = dataset.TransformerDataset(
        text=df_train[config.DATASET_TEXT].values,
        target=df_train[config.DATASET_LABEL].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )

    test_dataset = dataset.TransformerDataset(
        text=df_test[config.DATASET_TEXT].values,
        max_len=max_len,
        transformer=transformer
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        num_workers=config.TEST_WORKERS
    )

    device = config.DEVICE if config.DEVICE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, drop_out)
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / batch_size * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    # create obt for save preds class
    manage_preds = PredTools(df_test, transformer, drop_out, lr, batch_size, max_len)
    
    for epoch in range(1, config.EPOCHS+1):
        pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        f1_train = metrics.f1_score(targ_train,convert(pred_train), average='weighted')
        acc_train = metrics.accuracy_score(targ_train, convert(pred_train))
        
        pred_test = engine.test_fn(test_data_loader, model, device)
        
        # save epoch preds
        manage_preds.hold_epoch_preds(epoch, pred_test)
        
        df_new_results = pd.DataFrame({'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'accuracy_train':acc_train,
                            'f1-macro_train':f1_train,
                            'loss_train':loss_train,
                        }, index=[0]
        ) 
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f}".format(epoch, config.EPOCHS, f1_train, acc_train, loss_train))

    # save predicitons
    manage_preds.save_preds()
    
    return df_results

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    #rename old log files adding date YMD-HMS
    rename_logs()

    # load data
    df_train = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, sep='\t', index_col=0, nrows=config.N_ROWS)
    df_test = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TEST, sep='\t', index_col=0, nrows=config.N_ROWS)

    df_results = pd.DataFrame(columns=['epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'dropout',
                                        'accuracy_train',
                                        'f1-macro_train',
                                        'loss_train',
            ]
    )
    
    for transformer in tqdm(config.TRANSFORMERS, desc='TRANSFOMERS', position=0):
        for max_len in config.MAX_LEN:
            for batch_size in config.BATCH_SIZE:
                for drop_out in config.DROPOUT:
                    for lr in config.LR:
                        
                        tqdm.write(f'\nTransfomer: {transformer.split("/")[-1]} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr}')
                        
                        df_results = run(df_train,
                                            df_test, 
                                            max_len, 
                                            transformer, 
                                            batch_size, 
                                            drop_out,
                                            lr,
                                            df_results
                        )
                        
                        df_results.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '.tsv', index=False, sep='\t')