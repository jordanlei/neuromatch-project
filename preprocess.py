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


def indexby(x, var = "gocue", version = "wheel_velocity", left = 1, right = 3):
    index = int((np.round(x[var] + 500) / 10))
    return x[version][index - left: index + right]


def spike_preprocess(alldat):
    session_df = []
    trial_df = []

    spikes_df = []
    regions_df = []
    response_time_df = []
    go_cue_df = []
    feedback_time_df = []
    feedback_type_df = []
    contrast_left_df = []
    contrast_right_df = []

    mouse_name_df = []


    for i, dat in enumerate(alldat):
        print("session", i)
        trials = len(dat["gocue"])
        num_neurons = len(dat["spks"])
        spikes = np.concatenate([dat["spks"][:, i] for i in range(trials)])
        regions = np.concatenate([dat["brain_area"] for i in range(trials)]).tolist()
        response_time = np.repeat(dat["response_time"], num_neurons).tolist()

        gocue = np.repeat(dat["gocue"], num_neurons).tolist()
        feedback_time = np.repeat(dat["feedback_time"], num_neurons).tolist()
        feedback_type = np.repeat(dat["feedback_type"], num_neurons).tolist()
        contrast_left = np.repeat(dat["contrast_left"], num_neurons).tolist()
        contrast_right = np.repeat(dat["contrast_right"], num_neurons).tolist()
        
        trial = np.repeat(np.arange(trials), num_neurons).tolist()

        session_df += [i]*(trials * num_neurons)
        trial_df += trial
        mouse_name_df += [dat["mouse_name"]] * (trials * num_neurons)
        spikes_df += [spikes]
        regions_df += regions
        response_time_df += response_time
        go_cue_df += gocue
        feedback_time_df += feedback_time
        feedback_type_df += feedback_type
        contrast_left_df += contrast_left
        contrast_right_df += contrast_right

    mydict = {
    "session": session_df,
    "trial": trial_df,
    "region": regions_df,
    "response_time": response_time_df,
    "go_cue": go_cue_df,
    "feedback_time": feedback_time_df,
    "feedback_type": feedback_type_df,
    "contrast_left": contrast_left_df, 
    "contrast_right": contrast_right_df, 
    "mouse_name": mouse_name_df
    }

    df1 = pd.DataFrame(mydict)
    df2 = pd.DataFrame(np.concatenate(spikes_df, axis = 0))
    print("joining dataframes (may take up to a minute)...")
    df = df2.join(df1)
    
    regions = ["vis_ctx", "thal", "hipp", "other_ctx", "midbrain", "basal_ganglia", "cortical_subplate", "other"]
    brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                    ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                    ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                    ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                    ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                    ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                    ["BLA", "BMA", "EP", "EPd", "MEA"], # cortical subplate
                    [np.nan, ""]
                    ]
    
    region_dict = {}
    for i in range(len(regions)):
        region = regions[i]
        groups = brain_groups[i]
        for group in groups:
            region_dict[group] = region
    df["area"] = df["region"].apply(lambda x: region_dict[x] if x in region_dict.keys() else "other")
    print("done")

    return df

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
        s += 1

    go_trial = (contrast_left + contrast_right) != 0
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

        'delta_response_time': response_time_diff,

        'past_acc': pastify(pres_acc),
        'past_latency': pastify(latency),
        'past_difficulty': pastify(pres_difficulty),

        'fut_go_trial': futurify(go_trial).astype(bool),
        'fut_acc': futurify(pres_acc),
        'fut_latency': futurify(latency),
        'fut_difficulty': futurify(pres_difficulty),
    }

    dict_def = {
        'session': "session number, indexed 0 - 38",
        'session_type': "session type, train / test",
        'mouse_name': "name of the mouse",
        'trial_number': "trial number, indexed 0 to num_trials",
        'go_trial': "true if go_trial, false if no-go trial",

        'gocue': "time of go cue, in ms",
        'response_time': "time of response, in ms",
        'latency': "response time - go cue, in ms",
        'mouse_resp': "mouse response for a given trial",
        'wheel_velocity': "velocity of the wheel",
        'wheel_acceleration': "acceleration of the wheel (first derivative of velocity)",
        'feedback_onset': "feedback onset (reward or punishment)",
        'feedback_type': "-1 if punish, 0 if none, 1 if reward",
        'contrast_left': "left contrast",
        'contrast_right': "right contrast",
        'contrast_diff': "contrast left - contrast right",
        'pres_difficulty': "1 - absolute value(contrast_diff)",
        'pres_acc': "present accuracy, based on feedback",

        'delta_response_time': "first derivative of response time",

        'past_acc': "past accuracy, based on feedback",
        'past_latency': "past latency, in ms",
        'past_difficulty': "past difficulty",

        'fut_go_trial': "future go_trial, whether true or false",
        'fut_acc': "future accuracy, based on feedback",
        'fut_latency': "future latency, in ms",
        'fut_difficulty': "future difficulty",
    }


    df = pd.DataFrame(my_dict)
    df["zeros"] = 0

    df["gocue_vel_trial"] = df.apply(lambda x: indexby(x, var = "gocue", version = "wheel_velocity"), axis= 1)
    df["gocue_acc_trial"] = df.apply(lambda x: indexby(x, var = "gocue", version = "wheel_acceleration"), axis= 1)

    df["stim_vel_trial"] = df.apply(lambda x: indexby(x, var = "zeros", version = "wheel_velocity"), axis= 1)
    df["stim_acc_trial"] = df.apply(lambda x: indexby(x, var = "zeros", version = "wheel_acceleration"), axis= 1)

    df["rt_vel_trial"] = df.apply(lambda x: indexby(x, var = "response_time", version = "wheel_velocity"), axis= 1)
    df["rt_acc_trial"] = df.apply(lambda x: indexby(x, var = "response_time", version = "wheel_acceleration"), axis= 1)

    if verbose:
        for col in my_dict.keys():
            prcol = col + " "*100
            print("[%s]   \t%s\t%s"%(df[col].dtype, prcol[:20], dict_def[col]))

    return df


#plotting violinplot / scatter function, supports filtering
def plots(df, y = "fut_latency", features = ["pres_acc", "fut_latency"], filter_: dict= None, hue = None, title = None):
    if filter_ is not None:
        for key in filter_.keys():
            df = df[df[key] == filter_[key]]

    for feature in features:
        if df[feature].dtype in [float, int]:
            if len(list(set(df[feature].dropna()))) < 10:
                plt.figure()
                sns.violinplot(df[feature], df[y])
                plt.xlabel(feature)
                plt.ylabel(y)
            else:
                plt.figure()
                sns.scatterplot(feature, y, data = df, hue=hue, alpha= 0.7)
                plt.xlabel(feature)
                plt.ylabel(y)
            plt.title(title)

#plotting histogram function, supports filtering
def histograms(df, x, by: str, filter_:dict = None, title = None):
    if filter_ is not None:
        for key in filter_.keys():
            df = df[df[key] == filter_[key]]

    plt.figure()
    for val in set(df[by]):
        plt.hist(x = x, data = df[df[by]==val], density = True, alpha = 0.7, label = "%s: %s"%(by, val), bins = 15)
        plt.xlabel(x)

    plt.legend()
    plt.title(title)

######################################################################################################################


def pickle_data(dfs):
    with open('processed_data.pickle', 'wb') as f:
        pickle.dump(dfs, f)
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
