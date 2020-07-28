import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Create dummy confidence data
def dummy_conf(features = ["conf", "prev_conf",  "perf"], samples = 1000):
    """
    Returns: an np.array of shape (1000 x features)
    """

    # Generate specified features
    prev_conf = np.random.normal(0, 1, (samples, 1))
    perf = np.random.normal(0, 1, (samples, 1))
    exp_perf = 0.3 * prev_conf

    surprise = perf - exp_perf

     # Generate confidence from features
    conf = prev_conf + surprise + np.random.normal(0, 1, (samples, 1)) * 0.001
    arr = np.hstack([conf, prev_conf, perf])
    df = pd.DataFrame(arr, columns = features)

    return df


def plots(df, var = "conf"):
    for feature in df.columns:
        plt.figure()
        plt.scatter(df[feature], df[var])
        plt.xlabel(feature)
        plt.ylabel(var)

from sklearn.linear_model import LinearRegression
def get_model(x, y, version = "linear"):
    if version == "linear":
        model = LinearRegression().fit(x, y)
    return model

def trial_lat_diff(dfs, df_key):
    """Calculates the difference of the feature of interest between the first and last trial
    in a session with a certain difficulty.

    dfs: DataFrame, all the data
    df_key: feature of interest
    """
    difference = dict() # dictionary that stores one array for the differences for each difficulty
    for diff in np.unique(dfs['pres_difficulty']): # iterate over difficulty values
        difference[diff] = []

        for session in np.unique(dfs['session']): # iterate over sessions
            current_df = dfs[(dfs['session'] == session) & (dfs['pres_difficulty']==diff)]
            key_arr = current_df[df_key].to_numpy()
            try:
                difference[diff].append(key_arr[-1] - key_arr[0])
            except:
                difference[diff].append(np.nan)
        difference[diff] = np.array(difference[diff])

    return difference


def main():
    df = dummy_conf()
    plt.scatter(df['prev_conf'],df['conf'])
    print(df.head())
    df.describe()

if __name__ == '__main__':
    main()
