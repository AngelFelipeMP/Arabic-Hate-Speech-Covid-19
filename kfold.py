import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
from utils import PredTools, rename_logs

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


def run(df_train, df_val, max_len, transformer, batch_size, drop_out, lr, df_results, fold):
    
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

    val_dataset = dataset.TransformerDataset(
        text=df_val[config.DATASET_TEXT].values,
        target=df_val[config.DATASET_LABEL].values,
        max_len=max_len,
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        num_workers=config.VAL_WORKERS
    )

    device = config.DEVICE if config.DEVICE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, drop_out, number_of_classes=len(df_train[config.DATASET_LABEL].unique().tolist()))
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
    manage_preds = PredTools(df_val, transformer, drop_out, lr, batch_size, max_len)
    
    for epoch in range(1, config.EPOCHS+1):
        pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        f1_train = metrics.f1_score(targ_train, pred_train, average='weighted')
        acc_train = metrics.accuracy_score(targ_train, pred_train)
        
        pred_val, targ_val, loss_val = engine.eval_fn(val_data_loader, model, device)
        f1_val = metrics.f1_score(targ_val, pred_val, average='weighted')
        acc_val = metrics.accuracy_score(targ_val, pred_val)
        
        # save epoch preds
        manage_preds.hold_epoch_preds(pred_val, targ_val, epoch, fold)
        
        df_new_results = pd.DataFrame({'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'accuracy_train':acc_train,
                            'f1-macro_train':f1_train,
                            'loss_train':loss_train,
                            'accuracy_val':acc_val,
                            'f1-macro_val':f1_val,
                            'loss_val':loss_val
                        }, index=[0]
        ) 
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-macro_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(epoch, config.EPOCHS, f1_train, acc_train, loss_train, f1_val, acc_val, loss_val))

    # save a fold preds
    manage_preds.concat_fold_preds()
            
    # avg and save logs
    if fold == config.SPLITS:
        # save all folds preds "gridsearch"
        manage_preds.save_preds()
    
    
    return df_results

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    #rename old log files adding date YMD-HMS
    rename_logs()

    df_data = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, sep='\t', index_col=0, nrows=config.N_ROWS)
    # df_data = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, sep='\t', index_col=0)
    # print(df_data)
    # df_data = df_data.tail(config.N_ROWS).reset_index(drop=True)
    # print(df_data)
    
    skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)

    df_results = pd.DataFrame(columns=['epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'dropout',
                                        'accuracy_train',
                                        'f1-macro_train',
                                        'loss_train',
                                        'accuracy_val',
                                        'f1-macro_val',
                                        'loss_val'
            ]
    )
    
    inter = len(config.TRANSFORMERS) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS
    grid_search_bar = tqdm(total=inter, desc='GRID SEARCH', position=1)
    

    for transformer in tqdm(config.TRANSFORMERS, desc='TRANSFOMERS', position=0):
        for max_len in config.MAX_LEN:
            for batch_size in config.BATCH_SIZE:
                for drop_out in config.DROPOUT:
                    for lr in config.LR:
                        
                        for fold, (train_index, val_index) in enumerate(skf.split(df_data[config.DATASET_TEXT], df_data[config.DATASET_LABEL]),start=1):
                            df_train = df_data.loc[train_index]
                            df_val = df_data.loc[val_index]
                            
                            tqdm.write(f'\nTransfomer: {transformer.split("/")[-1]} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Fold: {fold+1}/{config.SPLITS}')
                            
                            df_results = run(df_train,
                                                df_val, 
                                                max_len, 
                                                transformer, 
                                                batch_size, 
                                                drop_out,
                                                lr,
                                                df_results,
                                                fold
                            )
                        
                            grid_search_bar.update(1)

                        
                        df_results = df_results.groupby(['epoch',
                                                        'transformer',
                                                        'max_len',
                                                        'batch_size',
                                                        'lr',
                                                        'dropout'], as_index=False, sort=False)['accuracy_train',
                                                                                            'f1-macro_train',
                                                                                            'loss_train',
                                                                                            'accuracy_val',
                                                                                            'f1-macro_val',
                                                                                            'loss_val'].mean()
                        
                        df_results.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '.tsv', index=False, sep='\t')