import numpy as np
import random
import pandas as pd

def train_test_split_evenSites(df, split_pct, seed):
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Get unique site identifiers
    sites = df['STAID'].unique()
    
    # Store splits here
    train_splits = []
    test_splits  = []
    
    for site in sites:
        temp_df = df[df['STAID'] == site]
        split_ind = int(np.floor((1-split_pct)*int(len(temp_df))))
        start_ind = np.random.randint(0, len(temp_df) - split_ind)
        end_ind = start_ind + split_ind
        test_df = temp_df.iloc[start_ind:end_ind, :]
        train_df = pd.concat([temp_df.iloc[:start_ind, :], temp_df.iloc[end_ind:, :]])
        train_splits.append(train_df)
        test_splits.append(test_df)

    # Zip lists together, shuffle them, then unzip them
    zipped_list = list(zip(train_splits, test_splits))
    random.shuffle(zipped_list)
    train_splits, test_splits = zip(*zipped_list)
    
    Train = pd.concat(train_splits)
    Test = pd.concat(test_splits)
    
    Train.drop("STAID", axis=1, inplace=True)
    Test.drop('STAID', axis=1, inplace=True)
    
    X_train = Train.drop('Q', axis=1)
    y_train = Train['Q']
    X_test = Test.drop('Q', axis=1)
    y_test = Test['Q']
    
    return X_train, y_train, X_test, y_test
