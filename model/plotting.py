"""File to put finalized versions of plotting functions"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
# Latency over time

def plot_over_trials(df, df_keys):
    """Plots the features of a single session over trials.

    df: DataFrame, data of a single session
    df_keys: List, Feature of intersts
    """
    for diff in np.unique(df['pres_difficulty']):
        plt.figure()
        plt.suptitle(f'Difficulty: {diff}')

        print('Trials with this difficulty: \n')
        print(np.where(df['pres_difficulty']==diff))

        for key in df_keys:
            plt.plot(df[df['pres_difficulty']==diff][key].to_numpy(), label = f"{key}")

        plt.legend(loc='best')
        plt.show()
