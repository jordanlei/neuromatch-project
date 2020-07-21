import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


# Create dummy confidence data
def dummy_conf(features = ["prev_conf",  "perf"], samples = 1000):
    """
    Returns: an np.array of shape (1000 x features)
    """

    # Generate specified features
    prev_conf = np.random.random((samples, 1))
    perf = np.random.random((samples, 1))
    exp_perf = prev_conf

    surprise = perf - exp_perf
    
     # Generate confidence from features
    conf = prev_conf + surprise + np.random.random((samples, 1)) * 0.001
    arr = np.hstack([conf, prev_conf, perf])
    df = pd.DataFrame(arr, columns = features)
    return df

def dummy_data2(alpha,
                beta,
                delta,
                features = ["conf", "prev_conf",  "acc"], 
                samples = 1000):
    """
    Returns: an np.array of shape (1000 x features)
    """
    # previous trial
    prev_conf = np.random.random((samples, 1))
    acc = np.random.random((samples, 1))
    
    #current trial
    exp_perf = alpha*prev_conf
    noise = (np.random.random((samples, 1)) * 0.001)
    conf = beta*prev_conf + delta*acc + noise
    arr = np.hstack([conf, prev_conf, acc])
    df = pd.DataFrame(arr, columns = features)
    
    return df
    




from sklearn.linear_model import LinearRegression
def model(x, y, version = "linear"):
    if version == "linear": 
        model = LinearRegression().fit(x, y)
    return model
    
    

def main():
    df = dummy_conf()
    print(df.head())
    df.describe()

if __name__ == '__main__':
    main()