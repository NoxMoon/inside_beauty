import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold

def oof_preds(X_train, y_train, model, folds=None, return_prob=True):
    if folds is None:
        folds = KFold(4, random_state=777)
        
    oof_preds = np.zeros(y_train.shape)
    oof_preds_proba = np.zeros(y_train.shape)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
        trn_x, val_x = X_train[trn_idx], X_train[val_idx]
        trn_y, val_y = y_train[trn_idx], y_train[val_idx]
        
        model.fit(trn_x, trn_y)
        oof_preds[val_idx] = model.predict(val_x)
        if return_prob and hasattr(model, 'predict_proba'):
            oof_preds_proba[val_idx] = model.predict_proba(val_x)
        
    if return_prob:
        return oof_preds, oof_preds_proba
    else:
        return oof_preds

def report_auc_by_class(y_test, y_pred_proba, classes=None):
   
    n_classes = y_test.shape[1]
    roc_auc = np.empty(n_classes)
    
    if classes is None:
        classes = ['class '+str(i+1) for i in range(n_classes)]

    plt.figure(figsize=(7,7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr, tpr)
        plt.plot(fpr, tpr,
              label='%s (auc = %0.2f)' % (classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.axis('equal')
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.legend(loc="lower right", bbox_to_anchor=(1.8,0))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.show()
    
    return roc_auc


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
    
    
def mean_encode(train, val, features_to_encode, target, drop=False, min_sample_leaf=10):
    train_encode = train.copy(deep=True)
    val_encode = val.copy(deep=True)
    for feature in features_to_encode:
        train_global_mean = train[target].mean()
        train_encode_map = pd.DataFrame(index = train[feature].unique())
        train_encode[feature+'_mean_encode'] = np.nan
        kf = KFold(n_splits=5, shuffle=False)
        for rest, this in kf.split(train):
            train_rest_global_mean = train[target].iloc[rest].mean()
            count = train.iloc[rest].groupby(feature)[target].count()
            encode_map = train.iloc[rest].groupby(feature)[target].mean()
            encode_map = (encode_map * count + train_rest_global_mean * min_sample_leaf)/(count + min_sample_leaf)
            encoded_feature = train.iloc[this][feature].map(encode_map).values
            train_encode[feature+'_mean_encode'].iloc[this] = train[feature].iloc[this].map(encode_map).values
            train_encode_map = pd.concat((train_encode_map, encode_map), axis=1, sort=False)
            train_encode_map.fillna(train_rest_global_mean, inplace=True) 
            train_encode[feature+'_mean_encode'].fillna(train_rest_global_mean, inplace=True)
            
        train_encode_map['avg'] = train_encode_map.mean(axis=1)
        val_encode[feature+'_mean_encode'] = val[feature].map(train_encode_map['avg'])
        val_encode[feature+'_mean_encode'].fillna(train_global_mean,inplace=True)
        #print('correlation',train[target].corr(train_encode[feature+'_mean_encode']))
        
    if drop: #drop unencoded features
        train_encode.drop(features_to_encode, axis=1, inplace=True)
        val_encode.drop(features_to_encode, axis=1, inplace=True)
    return train_encode, val_encode


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)