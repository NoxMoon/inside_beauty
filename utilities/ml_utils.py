import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

def oof_preds(X_train, y_train, model, folds=None):
    if folds is None:
        folds = KFold(4, random_state=777)
        
    oof_preds = np.zeros(y_train.shape)
    oof_preds_proba = np.zeros(y_train.shape)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
        trn_x, val_x = X_train[trn_idx], X_train[val_idx]
        trn_y, val_y = y_train[trn_idx], y_train[val_idx]
        
        model.fit(trn_x, trn_y)
        oof_preds[val_idx] = model.predict(val_x)
        oof_preds_proba[val_idx] = model.predict_proba(val_x)
        
    return oof_preds, oof_preds_proba

def multilable_confusion_matrix(y_true, y_pred, classes=None):
# confusion matrix for multilabel classification
# diagonal counts true positives
# off-diagonal counts false positives
    if (y_true.shape != y_pred.shape):
        raise ValueError("y_true and y_pred shape do not match in multilable_confusion_matrix")
        
    n_samples, n_class = y_true.shape
    m_confusion_matrix = np.zeros([n_class, n_class])
    
    for n in range(n_samples):
        set_true = set(np.where(y_true[n])[0])
        set_pred = set(np.where(y_pred[n])[0])
        
        tp = set_true.intersection(set_pred) 
        fp = set_pred.difference(set_true)
        #fn = set_true.difference(set_pred)
        
        for i in tp:
           m_confusion_matrix[i,i] += 1
        for i in set_true:
            for j in fp:
                m_confusion_matrix[i,j] += 1
                
    confusion_matrix_df = pd.DataFrame(m_confusion_matrix.astype('int'), index=classes, columns=classes)
    confusion_matrix_df.index.name = "true class"
    confusion_matrix_df.columns.name = "predicted class"
    return confusion_matrix_df

def hamming_score(y_true, y_pred, score_by_sample=False):
# intersection over union score for multilabel classification
    if (y_true.shape != y_pred.shape):
        raise ValueError("y_true and y_pred shape do not match in hamming_score")
        
    n_samples, n_class = y_true.shape
    acc_list = []
    for n in range(n_samples):
        set_true = set(np.where(y_true[n])[0])
        set_pred = set(np.where(y_pred[n])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
        
    if score_by_sample:
        return np.mean(acc_list), acc_list
    else:
        return np.mean(acc_list)