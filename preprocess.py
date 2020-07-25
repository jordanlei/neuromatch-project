import os, requests
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import pickle
from data.get_data import *


def train_test_split(df, test_set=[1, 4, 8, 12, 19, 22, 25, 30, 35]):
    """Filter out the test set from the data. DO NOT TOUCH THOSE.

    df: pd.DataFrame, dataframe to filter
    test_set: list, list of sessions to exclude from analysis
    returns: train_df,  test_df
    """
    train = df[df['session'].apply(lambda x: int(float(x)) not in test_set)]
    test = df[df['session'].apply(lambda x: int(float(x)) in test_set)]
    return train, test

def check_lengths(dfs):
    """Check the lenghts of the dataframes are equal
    
    dfs: list of pd.DataFrames
    """
    lengths = [len(df)for df in dfs]
    if len(set(lengths)) == 1:
        print('woohoo! all dfs have same len! you are chillin bb')
    else:
        raise Exception('Dataframes are not equal lengths!')

def futurify(arr): 
    arr = np.array(arr)
    return np.concatenate([arr.flatten()[1:], [np.nan]])

def pastify(arr):
    arr = np.array(arr)
    return np.concatenate([[np.nan], arr.flatten()[:-1]])

def preprocess(alldat, verbose = False):
    s = 0
    test_sessions = [1, 4, 8, 12, 19, 22, 25, 30, 35]

    # Initialize empty list 
    session = np.array([]) 
    session_type = np.array([]) 
    mouse_name = np.array([])   
    trial_number = np.array([]) 
    go_trial = np.array([])   #either TRUE (Go trial) or FALSE(= NoGo trial)
    stim_loc = np.array([]) 
    ideal_resp = np.array([]) 
    gocue = np.array([]) #dat['gocue']: when the go cue sound was played. 
    latency = np.array([]) 
    response_time = np.array([]) 
    response_time_diff = np.array([])
    mouse_resp = np.array([]) 
    wheel_velocity = [] 
    wheel_acceleration = []
    pres_acc = np.array([]) 
    feedback_onset = np.array([]) 
    feedback_type = np.array([]) 
    contrast_left = np.array([]) 
    contrast_right = np.array([]) 
    contrast_diff = np.array([]) 
    pres_difficulty = np.array([]) 

    # Loop to fill all lists with data from Steinmetz dataset
    for dat in alldat:
        s += 1
        num_trials = len(dat["gocue"])
        session = np.concatenate([session, [s]*num_trials])
        mouse_name = np.concatenate([mouse_name, [dat["mouse_name"]]*num_trials])
        trial_number = np.concatenate([trial_number, np.arange(num_trials)])
        contrast_left = np.concatenate([contrast_left, dat["contrast_left"].flatten()])
        contrast_right = np.concatenate([contrast_right, dat["contrast_right"].flatten()])

        gocue = np.concatenate([gocue, dat["gocue"].flatten() * 1000])
        response_time = np.concatenate([response_time, dat["response_time"].flatten() * 1000])
        response_time_diff = np.concatenate([response_time_diff, np.diff(dat["response_time"].flatten()), [np.nan]])

        mouse_resp = np.concatenate([mouse_resp, dat["response"].flatten()])

        wheel_velocity = wheel_velocity + dat["wheel"][0].tolist()
        
        wheel_acc = np.zeros(dat["wheel"][0].shape)* np.nan
        wheel_acc[:, :-1] = np.diff(dat["wheel"][0])

        wheel_acceleration = wheel_acceleration + wheel_acc.tolist()
        feedback_onset = np.concatenate([feedback_onset, dat["feedback_time"].flatten()])
        feedback_type = np.concatenate([feedback_type, dat["feedback_type"].flatten()])

    go_trial = contrast_left * contrast_right != 0
    session_type = ["test" if x in [1, 4, 8, 12, 19, 22, 25, 30, 35] else "train" for x in session]
    latency = response_time - gocue
    contrast_diff = contrast_left - contrast_right
    pres_difficulty = 1 - np.abs(contrast_diff)
    pres_acc = (feedback_type > 0).astype(int)

    my_dict = {'session': session, 
        'session_type': session_type,
        'mouse_name': mouse_name, 
        'trial_number': trial_number,
        'go_trial': go_trial,

        'gocue': gocue,
        'response_time': response_time,
        'latency': latency,
        'mouse_resp': mouse_resp,
        'wheel_velocity': wheel_velocity,
        'wheel_acceleration': wheel_acceleration,
        'feedback_onset': feedback_onset,
        'feedback_type': feedback_type,
        'contrast_left': contrast_left,
        'contrast_right': contrast_right,
        'contrast_diff': contrast_diff,
        'pres_difficulty': pres_difficulty,
        'pres_acc': pres_acc,

        'response_time_diff': response_time_diff,

        'past_acc': pastify(pres_acc), 
        'past_latency': pastify(latency), 
        'past_difficulty': pastify(pres_difficulty),

        'fut_go_trial': futurify(go_trial),
        'fut_acc': futurify(pres_acc),
        'fut_latency': futurify(latency),
        'fut_difficulty': futurify(pres_difficulty),
    }

    for i in my_dict.keys():
        print(i) 
        print(len(my_dict[i]))

    df = pd.DataFrame(my_dict)

    if verbose: 
        for col in df.columns:
            print("column %s dtype: "%(col, df[col].dtype))

    return df
######################################################################################################################


def pickle_data():
    alldat = load()
    dfs = pd.concat([preprocess(alldat, i) for i in range(39)])
    with open('processed_data.pickle', 'wb') as f:
        pickle.dump( dfs, f)
        print('Processed Data Dumped')

def load_processed_data():
    with open('processed_data.pickle', 'rb') as f:
        dfs = pickle.load(f)
        print("Processed Data loaded")
        return dfs

def main(): 
    download()
    alldat = load()
    print("Data Loaded. Proceeding to Preprocessing...")
    session = 12

    dfs = preprocess(alldat, verbose=True)
    
    print(dfs.shape)
    print(dfs.columns)
    print(dfs.head())

    train, test = train_test_split(dfs)

    

    print(train.shape)
    print(set(train["session"]))

    print(test.shape)
    print(set(test["session"]))

if __name__ == "__main__":
    main()
