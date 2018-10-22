from scipy import stats
import pandas as pd
import pickle
import numpy as np
from ast import literal_eval

def get_matching_ingredient_count(df):
# returns 1 if matching ingredient exist, 0 otherwise  
    pickle_in = open("../data_cleaning/ingredient_match_dict.pickle","rb")
    ingredient_names = list(set(pickle.load(pickle_in).values()))
    ingredient_count_df = pd.DataFrame(np.zeros([df.shape[0], len(ingredient_names)]), columns=ingredient_names)

    for i in range(df.shape[0]):
        ilist = list(set(literal_eval(df['inactive_ingredient_matched_list'].iloc[i])
                        +literal_eval(df['active_ingredient_matched_list'].iloc[i])))
        ingredient_count_df.iloc[i].loc[ilist] = 1
    
    cols = ingredient_count_df.sum()!=0
    ingredient_count_df = ingredient_count_df.loc[:, cols].astype('int8')    
    return ingredient_count_df


def get_ingredient_count(df):
# returns 1 if ingredient exist, 0 otherwise    
    pickle_in = open("../data_cleaning/ingredient_match_dict.pickle","rb")
    ingredient_names = list(set(pickle.load(pickle_in).keys()))
    ingredient_count_df = pd.DataFrame(np.zeros([df.shape[0], len(ingredient_names)]), columns=ingredient_names)

    for i in range(df.shape[0]):
        ilist = list(set(literal_eval(df['inactive_ingredient_list'].iloc[i])
                        +literal_eval(df['active_ingredient_list'].iloc[i])))
        ingredient_count_df.iloc[i].loc[ilist] = 1
        
    cols = ingredient_count_df.sum()!=0
    ingredient_count_df = ingredient_count_df.loc[:, cols].astype('int8') 
    return ingredient_count_df


def lr_stats(df, cols, y):
    slopes = []
    intercepts = []
    rs = []
    p_values = []
    stderrs = []
    for col in cols:
        mask = ~df[col].isnull() & ~df[y].isnull()
        slope, intercept, r, p_value, stderr = stats.linregress(df[col].loc[mask], df[y].loc[mask])   
        slopes.append(slope)
        intercepts.append(intercept)
        rs.append(r)
        p_values.append(p_value)
        stderrs.append(stderr)

    results = pd.DataFrame({'slope':slopes, 
                            'intercept':intercepts, 
                            'r-value': rs,
                            'p-value': p_values,
                            'stderr': stderrs}, index=cols)

    results = results.sort_values('p-value')
    return results

def chi2_contingency(df, cols, y):
    chi2s = []
    p_values = []
    for col in cols:
        observed = df.groupby(col)[y].value_counts().unstack(level=-1)
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        chi2s.append(chi2)
        p_values.append(p_value)
        
    results = pd.DataFrame({'chi2': chi2s,
                            'p-value': p_values}, index=cols)  
    results = results.sort_values('p-value')
    return results


def chi2_square(df, cols, y):
    chi2s = []
    p_values = []
    counts = []
    for col in cols:
        observed = df.loc[df[col]==1][y].value_counts()
        chi2, p_value = stats.chisquare(observed)
        chi2s.append(chi2)
        p_values.append(p_value)
        counts.append(observed.sum())
        
    results = pd.DataFrame({'chi2': chi2s,
                            'p-value': p_values,
                            'counts': counts}, index=cols)  
    results = results.sort_values('p-value')
    return results