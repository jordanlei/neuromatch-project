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


def plots(df): 
    for feature in df.columns: 
        plt.figure()
        plt.scatter(df[feature], df["conf"])  
        plt.xlabel(feature)
        plt.ylabel("conf") 

from sklearn.linear_model import LinearRegression
def get_model(x, y, version = "linear"):
    if version == "linear": 
        model = LinearRegression().fit(x, y)
    return model
    
    

def main():
    df = dummy_conf()
    plt.scatter(df['prev_conf'],df['conf'])
    print(df.head())
    df.describe()

if __name__ == '__main__':
    main()