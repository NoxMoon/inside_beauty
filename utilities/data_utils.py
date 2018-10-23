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