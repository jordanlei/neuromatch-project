

from matplotlib import rcParams 
from matplotlib import pyplot as plt
from scipy.special import expit
import numpy as np

import os, requests

# Retrieve all data
fname = []
for j in range(3):
  fname.append('steinmetz_part%d.npz'%j)
url = ["https://osf.io/agvxh/download"]
url.append("https://osf.io/uv3mw/download")
url.append("https://osf.io/ehmw2/download")

for j in range(len(url)):
  if not os.path.isfile(fname[j]):
    try:
      r = requests.get(url[j])
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname[j], "wb") as fid:
          fid.write(r.content)

# Load all data
alldat = np.array([])
for j in range(len(fname)):
  alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

# Grab specific session
dat = alldat[11]
print(dat.keys())

# -------------------------- FEATURES -------------------------- 

alpha = 1
total_time = 2 # in s

conf_pred = np.empty([len(dat['response_time']),])
surprise = np.empty([len(dat['response_time']),])
expected_performance = np.empty([len(dat['response_time']),])
# confidence = np.squeeze(1 - ((dat['response_time'])/total_time))
confidence = np.squeeze(1 - ((dat['response_time']-dat['gocue'])/total_time))
actual_performance = (dat['feedback_type']> 0).astype(float)
difficulty = 1 - np.abs(dat['contrast_right']-dat['contrast_left'])

# Set initial confidence
conf_pred[0] = confidence[0]

# Convert into trial-feature matrix
past_confidence = confidence[:-1]
current_confidence = confidence[1:]

# -------------------------- MODEL -------------------------- 

# Compute predicted confidence
for t in np.arange(len(confidence)-1):
    expected_performance[t+1] = (1-difficulty[t+1])*conf_pred[t]
    surprise[t+1] = actual_performance[t+1] - expected_performance[t+1]
    conf_pred[t+1] = expit(conf_pred[t] + alpha*(surprise[t+1]))

# -------------------------- RESULTS -------------------------- 

# Plot the model results with ground truth
plt.figure(num=None, figsize=(12, 2), dpi=80, facecolor='w', edgecolor='k')
plt.plot(conf_pred,color='blue')
plt.plot(confidence,color='black')
plt.ylim([0,1]);