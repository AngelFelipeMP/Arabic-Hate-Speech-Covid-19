import pandas as pd
import config
from sklearn import metrics

class Ensemble:
    def __init__(self):
        self.file_preds = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '_predictions' +'.tsv'
        self.file_metrics = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '.tsv'
        
    
    def create(self):
        df_metrics = pd.read_csv(self.file_metrics, sep='\t')
        
        if any(ensemble in df_metrics['transformer'].values for ensemble in ['Highest_sum', 'Majority_vote']):
            print('Ensemble already created')
        else:
            self.col_ensembles()
            self.results_ensemble()
            print('Ensemble created')
    
    def col_ensembles(self):
        self.df_preds = pd.read_csv(self.file_preds, sep='\t')
        
        for epoch in range(1, config.EPOCHS+1):
            # create ensemble
            epoch_cols = [col for col in self.df_preds.columns if col.endswith(str(epoch))]
            
            self.df_preds['Majority_vote' + '_' + str(epoch)] = self.df_preds.loc[:, epoch_cols].apply(lambda x: 1 if sum(self.get_vote(x)) > len(epoch_cols)/2 else 0, axis=1)
            self.df_preds['Highest_sum' + '_' + str(epoch)] = self.df_preds.loc[:, epoch_cols].apply(lambda x: 1 if x.sum()/len(epoch_cols) > 0.5 else 0, axis=1)
            
        # save ensemble preds
        self.df_preds.to_csv(self.file_preds, index=False, sep='\t')
        
    def get_vote(self,scores):
        return [1 if sc > 0.5 else 0 for sc in scores]
        
    
    def calculate(self, ensemble, epoch):
        f1_val= []
        acc_val= []
        
        for fold in range(1, config.SPLITS+1):
            pred_val = self.df_preds.loc[ self.df_preds['fold']==fold, [ensemble + '_' + str(epoch)]].values
            targ_val = self.df_preds.loc[ self.df_preds['fold']==fold, ['target']].values
            
            f1_val.append(metrics.f1_score(targ_val, pred_val, average='weighted'))
            acc_val.append(metrics.accuracy_score(targ_val, pred_val))
            
        return sum(f1_val)/len(f1_val), sum(acc_val)/len(acc_val)

    
    def results_ensemble(self):
        df_metrics = pd.read_csv(self.file_metrics, sep='\t')
        
        for epoch in range(1, config.EPOCHS+1):
            for ensemble in ['Majority_vote', 'Highest_sum']:
            
                f1_val, acc_val = self.calculate(ensemble, epoch)
            
                df_new_row = pd.DataFrame({'epoch':epoch,
                                    'transformer':ensemble,
                                    'max_len':None,
                                    'batch_size':None,
                                    'lr':None,
                                    'dropout':None,
                                    'accuracy_train':None,
                                    'f1-macro_train':None,
                                    'loss_train':None,
                                    'accuracy_val':acc_val,
                                    'f1-macro_val':f1_val,
                                    'loss_val':None
                                }, index=[0]
                ) 
                
                df_metrics = pd.concat([df_metrics, df_new_row], ignore_index=True)
            
        df_metrics.to_csv(self.file_metrics, index=False, sep='\t')
        
        
if __name__ == "__main__":
    ensembles = Ensemble()
    ensembles.create()