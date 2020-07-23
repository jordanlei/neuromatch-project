import os, requests
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
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


# Define trial_type_func, a function to determine trial_type from session's contrast columns
def trial_type_func (row):
    if row['contrast_left'] == 0 and row['contrast_right'] != 0:
        trial_type = 'A'
        return trial_type
    elif row['contrast_left'] != 0 and row['contrast_right'] == 0:
        trial_type = 'B'
        return trial_type
    elif row['contrast_right'] or row['contrast_left'] != 0 and row['contrast_right'] > row['contrast_left']:
        trial_type = 'C'
        return trial_type
    elif row['contrast_right'] or row['contrast_left'] != 0 and row['contrast_right'] < row['contrast_left']:
        trial_type = 'D'
        return trial_type
    elif row['contrast_left'] == 0 and row['contrast_right'] == 0:
        trial_type = 'E'
        return trial_type
    else:
        return 'NaN'

# Define stim_loc_func function to take in session's contrast columns and determine stimulus location for the trial (row or index pos)
def stim_loc_func (row):
    if row['contrast_left'] == 0:                                                                              # stim on right
        stim_loc = -1.
        return stim_loc
    elif row['contrast_right'] == 0:                                                                           # stim on left
        stim_loc = 1.
        return stim_loc
    elif row['contrast_right'] or row['contrast_left'] != 0 and row['contrast_right'] > row['contrast_left']:
        stim_loc = 2.
        return stim_loc
    elif row['contrast_right'] or row['contrast_left'] != 0 and row['contrast_right'] < row['contrast_left']:
        stim_loc = 2.
        return stim_loc
    elif row['contrast_left'] == 0 and row['contrast_right'] == 0:
        stim_loc = 0.
        return stim_loc
    else:
        return 'NaN'

# Define winning_stim_func to take in session's contrast columns and determine side of higher/winning stimulus contrast for each trial(row or index pos)
def winning_stim_func (row):
    if row['contrast_left'] == 0 and row['contrast_right'] != 0:
        winning_stim = {'right': -1.}
        return winning_stim
    elif row['contrast_right'] == 0 and row['contrast_left'] != 0:
        winning_stim = {'left': 1.}
        return winning_stim
    elif row['contrast_right'] or row['contrast_left'] != 0 and row['contrast_right'] > row['contrast_left']:
        winning_stim = {'right': -1.}
        return winning_stim
    elif row['contrast_right'] or row['contrast_left'] != 0 and row['contrast_right'] < row['contrast_left']:
        winning_stim = {'left': 1.}
        return winning_stim
    elif row['contrast_left'] == 0 and row['contrast_right'] == 0:
        winning_stim = {'no-go': 0.}
        return winning_stim
    else:
        return 'NaN'

# Define trial_ideal_func, a function to determine what the ideal response would be given trial type
# Accuracy integers defined by Steinmetz
def trial_ideal_func (row):
    if row['trial_type'] == 'A': 
        correct_resp = -1.
        return correct_resp 
    elif row['trial_type'] == 'B':
        correct_resp = 1.
        return correct_resp 
    elif row['trial_type'] == 'C':
        correct_resp = -1.
        return correct_resp
    elif row['trial_type'] == 'D':
        correct_resp = 1.
        return correct_resp
    elif row['trial_type'] == 'E':
        correct_resp = 0.
        return correct_resp
    else:
        return 'error in trial_acc_func'

# Define trial_acc_func, a function to determine if mouse was right(1) or wrong(0) on the present trial
def trial_acc_func (row):
    if row['mouse_resp'] == row['ideal_resp']: 
        m_acc = 1.
        return m_acc 
    else:
        m_acc = 0.
        return m_acc

# Define trial_difficulty, a function to determine the difficulty for the present/current trial (1 - abs(left - right contrast))
def trial_difficulty(row):
    difficulty = 1 - np.abs(row['contrast_left'] - row['contrast_right'])
    return difficulty

def trial_contrast_diff (row):
    contrast_diff = row['contrast_left'] - row['contrast_right']
    return contrast_diff 

# Define trial_contrast_abs_diff, a function to assess the absolute value of contrast difference between left and right stimulus on a given trial (even when just one appears)
def trial_contrast_abs_diff (row):
    abs_contrast_diff = np.abs(row['contrast_left'] - row['contrast_right'])
    return abs_contrast_diff 

# Define latency_func, a function to determine the latency between rt and go cue 
def latency_func(row):
    latency = row['resp_time'] - row['gocue_onset']
    return latency
   
def preprocess(alldat, session):
    print("Session No: %s"%(session))
    dat = alldat[session]
    empty = np.empty([len(dat['gocue']), 1])
    for i in range(len(empty)):
        empty[i] = session

    session_df = pd.DataFrame(empty)
    session_df = session_df.rename(columns={0: 'session'}).astype(int)

    # Create 1 column df of len (quantity of trials in that session) filled with session's mouse name
    empty_b = np.empty([len(dat['gocue']), 1]).astype(str)

    for i in range(len(empty_b)):
        mouse_name = alldat[session]['mouse_name']
        empty_b[i] = mouse_name

    mouse_name_df = pd.DataFrame(empty_b)
    mouse_name_df = mouse_name_df.rename(columns={0: 'mouse_name'})

    # Create first DataFrame to apped all desired data to (should have session # and mouse name to start, where each row is a trial)
    foundation = pd.concat([session_df, mouse_name_df], axis = 1)
    foundation.head()

    # Make a list to define range in for loop 
    # Session columns related to stimulus contrast: dat['contrast_right'] and dat['contrast_left'] hold contrast values (0, .25, .5, 1.0) 
    contrast_trials = {'contrast_right': dat['contrast_right'], 'contrast_left': dat['contrast_left']}
    contrast_trials = pd.DataFrame(contrast_trials)


    # Apply trial_type_func to session's contrast columns to determine trial type of each trial (row or index pos)
    trial_type = contrast_trials.apply(lambda row: trial_type_func(row), axis=1)
    trial_type = pd.DataFrame(trial_type)
    trial_type = trial_type.rename(columns={0: 'trial_type'})

    # Apply stim_loc_func to session's contrast columns to determine stimulus location for the trial (row or index pos)
    stim_loc = contrast_trials.apply(lambda row: stim_loc_func(row), axis=1)
    stim_loc = pd.DataFrame(stim_loc)
    stim_loc = stim_loc.rename(columns={0: 'stim_loc'})

    # Apply winning_stim_func to session's contrast columns to determine side of higher/winning stimulus contrast for each trial(row or index pos)
    winning_stim = contrast_trials.apply(lambda row: winning_stim_func(row), axis=1)
    winning_stim = pd.DataFrame(winning_stim)
    winning_stim = winning_stim.rename(columns={0: 'winning_stim'})

    # Concat created dfs into one dataframe (s1 = Step 1)
    # trials_s1 = every row is a trial and this has every column associated with Step 1 (trial type and stim information)
    trials_s1 = pd.concat([foundation, trial_type, contrast_trials, stim_loc, winning_stim], axis = 1)

    ######################################################################################################################
    # Step 2: Extracting accuracy information for every trial in the session
    #Accuracy (ideal_resp, acc_prev, acc_pres)

    # Accuracy of present trial (t0) (acc_pres)
    # Make a list to define range in for loop 
    m_resp_list = {'mouse_resp': dat['response']}
    m_resp_df = pd.DataFrame(m_resp_list)

    # Apply trial_ideal_func to determine what a 'correct' or 'ideal' response would be on the present trial
    ideal_resp = trials_s1.apply(lambda row: trial_ideal_func(row), axis=1)
    ideal_resp = pd.DataFrame(ideal_resp)
    ideal_resp = ideal_resp.rename(columns={0: 'ideal_resp'})
    trials_s2_a= pd.concat([trials_s1, m_resp_df, ideal_resp], axis = 1)

    # Apply trial_acc_func to determine if mouse was correct in present trial
    pres_acc = trials_s2_a.apply(lambda row: trial_acc_func(row), axis=1)
    pres_acc = pd.DataFrame(pres_acc)
    pres_acc = pres_acc.rename(columns={0: 'pres_acc'})

    # Combine all dfs so far into one df for further investigation
    trials_s2_b= pd.concat([trials_s2_a, pres_acc], axis = 1)

    # Previous accuracy (acc_prev)
    # Create an empty array to assign previous accuracy values to
    empty = np.empty([len(dat['response']), 1])
    empty[0] = 'NaN' #ignore first index because there is no previous trial to this index

    for i in range(len(empty)-1):
        empty[i+1] = trials_s2_b['pres_acc'][i]

    prev_acc = pd.DataFrame(empty)
    prev_acc = prev_acc.rename(columns={0: 'prev_acc'})

    # Combine all new dfs into one (s2 = Step 2)
    trials_s2= pd.concat([trials_s2_a, prev_acc, pres_acc], axis = 1)
    

    ######################################################################################################################
    # Step 3: Assessing contrast differences and difficulty (contrast_diff, abs_contrast_diff, prev_difficulty, pres_difficulty)
    # Apply trial_contrast_diff to determine contrast difference on every trial (contrast_diff)
    contrast_diff = trials_s2.apply(lambda row: trial_contrast_diff(row), axis=1)
    contrast_diff = pd.DataFrame(contrast_diff)
    contrast_diff = contrast_diff.rename(columns={0: 'contrast_diff'})

    # Apply trial_contrast_abs_diff to determine absolute contrast difference on every trial (abs_contrast_diff)
    abs_contrast_diff = trials_s2.apply(lambda row: trial_contrast_abs_diff(row), axis=1)
    abs_contrast_diff = pd.DataFrame(abs_contrast_diff)
    abs_contrast_diff = abs_contrast_diff.rename(columns={0: 'abs_contrast_diff'})


    # Combine dataframes up to this point into one for further assessment
    trials_s3_a = pd.concat([trials_s2, contrast_diff, abs_contrast_diff], axis = 1)




    # Apply trial_difficulty to append a column holding present/current trial difficulty (pres_difficulty)
    pres_difficulty = trials_s3_a.apply(lambda row: trial_difficulty(row), axis=1)
    pres_difficulty = pd.DataFrame(pres_difficulty)
    pres_difficulty = pres_difficulty.rename(columns={0: 'pres_difficulty'})


    # Combine dataframes up to this point again for further assessment
    trials_s3_b = pd.concat([trials_s3_a, pres_difficulty], axis = 1)


    # Determine the difficulty from previous trial for every row (prev_difficulty)
    # Create an empty array to fill in with previous difficulty scores for every row (trial)
    empty = np.empty([len(dat['response']), 1])
    empty[0] = 'NaN' #ignore first index because there is no previous trial to this index

    for i in range(len(empty)-1):
        empty[i+1] = trials_s3_b['pres_difficulty'][i]

    prev_difficulty = pd.DataFrame(empty)
    prev_difficulty = prev_difficulty.rename(columns={0: 'prev_difficulty'})

    # Combine all dfs into one for a final Step 3 (s3) DataFrame
    trials_s3 = pd.concat([trials_s3_b, prev_difficulty], axis = 1)




    ######################################################################################################################
    # Step 4: Crucial time points
    # Time points (go_onset, resp_time)

    # Go cue onset
    go_onset = pd.DataFrame(dat['gocue'])
    go_onset = go_onset.rename(columns={0: 'gocue_onset'})

    # Response time
    resp_time = pd.DataFrame(dat['response_time'])
    resp_time = resp_time.rename(columns={0: 'resp_time'})

    # Combine dataframes into one cumulative df adding s4 (step 4)
    trials_s4_a = pd.concat([trials_s3, go_onset, resp_time], axis = 1)

    # Latency calculation


    # Apply trial_difficulty to append a column holding present/current trial difficulty (pres_difficulty)
    latency = trials_s4_a.apply(lambda row: latency_func(row), axis=1)
    latency = pd.DataFrame(latency)
    latency = latency.rename(columns={0: 'latency'})

    # Combine dataframes into one cumulative df adding s4 (step 4)
    trials_s4 = pd.concat([trials_s4_a, latency], axis = 1)


    trials_s4.head()


    ######################################################################################################################
    # Step 6: Extract information about delivered feedback (feedback_onset, feedback_type, prev_feedback)

    # Feedback onset time
    feedback_onset = pd.DataFrame(dat['feedback_time'])
    feedback_onset = feedback_onset.rename(columns={0: 'feedback_onset'})


    # Feedback type
    feedback_type = pd.DataFrame(dat['feedback_type'])
    feedback_type = feedback_type.rename(columns={0: 'feedback_type'})


    # Combine new dfs up to this point into one df for further exploration
    trials_s6_a = pd.concat([feedback_onset, feedback_type], axis =1)

    # Previous feedback

    # Create an empty array to fill in with previous difficulty scores for every row (trial)
    empty = np.empty([len(dat['response']), 1])
    empty[0] = 'NaN' #ignore first index because there is no previous trial to this index

    for i in range(len(empty)-1):
        empty[i+1] = trials_s6_a['feedback_type'][i]

    prev_feedback = pd.DataFrame(empty)
    prev_feedback = prev_feedback.rename(columns={0: 'prev_feedback'})

    # Combine all dfs into one for a final Step 3 (s3) DataFrame
    trials_s6 = pd.concat([trials_s4, trials_s6_a, prev_feedback], axis = 1)

    return trials_s6

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

    dfs = pd.concat([preprocess(alldat, i) for i in range(39)])
    
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
