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
        wheel_vel = [dat["wheel"][0][t] for t in range(num_trials)]
        wheel_velocity = wheel_velocity + wheel_vel
        wheel_acceleration = wheel_acceleration + wheel_vel
        feedback_onset = np.concatenate([feedback_onset, dat["feedback_time"].flatten()])
        feedback_type = np.concatenate([feedback_type, dat["feedback_type"].flatten()])

    go_trial = contrast_left * contrast_right != 0
    print(session.shape)
    session_type = ["test" if x in [1, 4, 8, 12, 19, 22, 25, 30, 35] else "train" for x in session]
    latency = response_time - gocue
    contrast_diff = contrast_left - contrast_right
    pres_difficulty = 1 - np.abs(contrast_diff)
    pres_acc = (feedback_type > 0).astype(int)

  


        # for t in range(len(dat.get('gocue'))): #just because the length of this field = number of trials for a given session
            # session #


            # if s == 1 or s== 4 or s == 8 or s == 12 or s == 19 or s == 22 or s== 25 or s == 30 or s == 35:
            #     session_type.append('test')
            # else: 
            #     session_type.append('train') 
            
            # mouse name
            # mouse_name.append(dat.get('mouse_name')) #you know... just the name of the little fellow
            
            # trial number
            # trial_number.append(t+1)      #trial number for that particular session

            # whether it was a go trial or not
            # go_trial.append((dat.get('contrast_left')[t] != 0) or (dat.get('contrast_right')[t] != 0))
            
            # trial type, stimulus location on pres trial, and ideal response
            # if dat.get('contrast_left')[t] == 0 and dat.get('contrast_right')[t] == 0:
            #     trial_t= 'F'
            # elif dat.get('contrast_left')[t] == 0:
            #     trial_t = 'A'
            #     stim_l = -1
            #     ideal_r = -1
            # elif dat.get('contrast_right')[t] == 0:
            #     trial_t = 'B'
            #     stim_l = 1
            #     ideal_r = 1
            # elif dat.get('contrast_left')[t] == dat.get('contrast_right')[t] :
            #     trial_t = 'C'
            #     stim_l = 2
            #     ideal_r = np.nan
            # elif dat.get('contrast_right')[t] > dat.get('contrast_left')[t] :
            #     trial_t = 'D'
            #     stim_l = 2
            #     ideal_r = -1
            # elif dat.get('contrast_right')[t] < dat.get('contrast_left')[t] :
            #     trial_t = 'E'
            #     stim_l= 2
            #     ideal_r = 1
            # else:
            #     trial_t = np.nan
            #     stim_l = np.nan
            #     ideal_r = np.nan
                
            # trial_type.append(trial_t) 
            # stim_loc.append(stim_l)
            # ideal_resp.append(ideal_r)
            
            # go cue onset
            # gocue.append(dat.get('gocue')[t][0]*1000) #the sencond index value is just there so that we can directly access floats from the DF
            
            # # mouse response time 
            # response_time.append(dat.get('response_time')[t][0]*1000)
            
            # # latency 
            # latency.append(dat.get('response_time')[t][0]*1000 - dat.get('gocue')[t][0]*1000)

            # append mouse's response
            # mouse_resp.append(dat.get('response')[t])
            
            # velocity
            # wheel_velocity.append(dat['wheel'][0][t] * .135) 
            
            # wheel acceleration
            # wheel_acceleration.append(np.diff(wheel_velocity[t]/10))
        
            # feedback onset 
            # feedback_onset.append((dat.get('feedback_time')[t][0]*1000))
            
            # feedback type 
            # feedback_type.append(int(dat.get('feedback_type')[t])) #feedback type: positive (+1) means reward, negative (-1) means white noise burst 
        
            # stimulus contrasts and differences between them
            # contrast_left.append(dat.get('contrast_left')[t])
            # contrast_right.append(dat.get('contrast_right')[t])
            
            # if stim_loc[t] != 0:
            #     contrast_diff.append((dat.get('contrast_left')[t] - dat.get('contrast_right')[t]).astype(float))
            # else:
            #     contrast_diff.append(np.nan)

            # present difficulty
            # if (dat.get('contrast_left')[t] != 0) or (dat.get('contrast_right')[t] != 0):
            #     pres_difficulty.append(1-np.abs(dat.get('contrast_left')[t]-dat.get('contrast_right')[t]))
            # else:
            #     pres_difficulty.append(np.nan)
                
            # # mouse's accuracy on present trial
            # if dat.get("feedback_type")[t] == 1: 
            #     mouse_acc_binary = 1
            #     pres_acc.append(mouse_acc_binary)
            # else:
            #     mouse_acc_binary = 0
            #     pres_acc.append(mouse_acc_binary)

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
