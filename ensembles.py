import pandas as pd
import config
from sklearn import metrics
import argparse
import warnings
warnings.filterwarnings('ignore')

class Ensemble:
    def __init__(self, domain):
        self.domain = domain
        self.file_preds = config.LOGS_PATH + '/' + self.domain + '_predictions' +'.tsv'
        self.file_metrics = config.LOGS_PATH + '/' + self.domain + '.tsv'
        self.ensembles = ['Majority_vote', 'Highest_sum']

    def create(self):
        ''''Run function'''
        self.df_preds = pd.read_csv(self.file_preds, sep='\t')
        
        if self.df_preds.columns[-1].startswith('Highest_sum') or self.df_preds.columns[-1].startswith('Majority_vote'):
            print('Ensemble already created')
        else:
            self.epoch_ensembles()
            self.best_models()
            self.best_ensembles()
            if 'test' not in self.domain:
                self.results_ensemble()
            print('Ensemble created')
    
    def epoch_ensembles(self):
        '''create ensemble for each epoch'''
        
        for epoch in range(1, config.EPOCHS+1):
            # create ensemble
            epoch_cols = [col for col in self.df_preds.columns if col.endswith(str(epoch))]
            
            self.df_preds['Majority_vote' + '_' + str(epoch)] = self.df_preds.loc[:, epoch_cols].apply(lambda x: 1 if sum(self.get_vote(x)) > len(epoch_cols)/2 else 0, axis=1)
            self.df_preds['Highest_sum' + '_' + str(epoch)] = self.df_preds.loc[:, epoch_cols].apply(lambda x: 1 if x.sum()/len(epoch_cols) > 0.5 else 0, axis=1)
            
        # save ensemble preds
        self.df_preds.to_csv(self.file_preds, index=False, sep='\t')
    
    def best_models(self):
        ''''create a df with metrics of the best models in the validation data'''
        df_cross_validation = pd.read_csv(config.LOGS_PATH + '/' + 'cross_validation.tsv', sep='\t')
        df_cross_validation = df_cross_validation.sort_values('f1-score_val', ascending=False)
        df_best_models = df_cross_validation.drop_duplicates(subset=['transformer'], keep='first')
    
        # grab bast model/epoch list
        tuple_list = list(zip(df_best_models['epoch'], df_best_models['transformer']))
        self.list_best_models = [(epo, trans.split('/')[-1]) for epo, trans in tuple_list if trans not in self.ensembles]
        
        # save tsv best models
        df_best_models.to_csv(config.LOGS_PATH + '/' + 'cross_validation_best_models.tsv', index=False, sep='\t')

    def get_pred_best_models(self):
        ''''extract predictions (from the best models)'''
        columns =  [col for col in self.df_preds.columns for epoch, trans in self.list_best_models if trans in col and col.endswith(str(epoch))]
        self.best_columns = [*self.df_preds.columns[:2], *columns]
        self.df_ensembles = self.df_preds.loc[:,self.best_columns]
        
    def best_ensembles(self):
        ''''create ensembles with the predictions of the best models'''
        self.df_preds = pd.read_csv(self.file_preds, sep='\t')
        self.get_pred_best_models()
        
        self.df_preds['Majority_vote'] = self.df_ensembles.loc[:, self.best_columns[2:]].apply(lambda x: 1 if sum(self.get_vote(x)) > len(x)/2 else 0, axis=1)
        self.df_preds['Highest_sum'] = self.df_ensembles.loc[:, self.best_columns[2:]].apply(lambda x: 1 if x.sum()/len(x) > 0.5 else 0, axis=1)
            
        # save ensemble preds
        self.df_preds.to_csv(self.file_preds, index=False, sep='\t')
        
    def get_vote(self,scores):
        return [1 if sc > 0.5 else 0 for sc in scores]
    
    def calculate(self, ensemble, epoch):
        ''''calculate metrics for ensembles'''
        f1_val, acc_val, recall_val, prec_val= [], [], [], []
        
        for fold in range(1, config.SPLITS+1):
            ensemble_col = ensemble if epoch == 'best' else ensemble + '_' + str(epoch)
            pred_val = self.df_preds.loc[ self.df_preds['fold']==fold, [ensemble_col]].values
            targ_val = self.df_preds.loc[ self.df_preds['fold']==fold, ['target']].values
            
            f1_val.append(metrics.f1_score(targ_val, pred_val, average='binary'))
            acc_val.append(metrics.accuracy_score(targ_val, pred_val))
            prec_val.append(metrics.precision_score(targ_val, pred_val, average='binary'))
            recall_val.append(metrics.recall_score(targ_val, pred_val, average='binary'))
            
        return sum(f1_val)/len(f1_val), sum(acc_val)/len(acc_val), sum(prec_val)/len(prec_val), sum(recall_val)/len(recall_val)

    
    def results_ensemble(self):
        '''add ensemble metrics to metric table'''
        df_metrics = pd.read_csv(self.file_metrics, sep='\t')
        
        for epoch in [l for l in range(1, config.EPOCHS+1)] + ['best']:
            for ensemble in ['Majority_vote', 'Highest_sum']:
            
                f1_val, acc_val, prec_val, recall_val = self.calculate(ensemble, epoch)
            
                df_new_row = pd.DataFrame({'epoch':epoch,
                                    'transformer':ensemble,
                                    'max_len':None,
                                    'batch_size':None,
                                    'lr':None,
                                    'dropout':None,
                                    'accuracy_train':None,
                                    'f1-score_train':None,
                                    'loss_train':None,
                                    'accuracy_val':acc_val,
                                    'f1-score_val':f1_val,
                                    'precision_val':prec_val,
                                    'recall_val':recall_val,
                                    'loss_val':None
                                }, index=[0]
                ) 
                
                df_metrics = pd.concat([df_metrics, df_new_row], ignore_index=True)
            
        df_metrics.to_csv(self.file_metrics, index=False, sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, help="Must be True or False", action='store_true')
    parser.add_argument("--kfold", default=False, help="Must be True or False", action='store_true')
    args = parser.parse_args()

    if args.test:
        domain = config.DOMAIN_TEST 
    elif args.kfold:
        domain = config.DOMAIN_CROSS_VALIDATION
    else:
        raise Exception('Please specify the domain: test or kfold')

    ensembles = Ensemble(domain)
    ensembles.create()