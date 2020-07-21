import os, requests
import numpy as np

def download():
    """
    returns: an np.array of all data (alldat)
    """  
    print("Downloading data. This may take a while ...")
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

def load():
    """
    returns: an np.array of all data (alldat)
    """ 
    print("Loading data. Patience, padawan ...")
    fname = []
    for j in range(3):
        fname.append('steinmetz_part%d.npz'%j)
        url = ["https://osf.io/agvxh/download"]
        url.append("https://osf.io/uv3mw/download")
        url.append("https://osf.io/ehmw2/download")

    alldat = np.array([])
    for j in range(len(fname)):
        alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))
    return alldat