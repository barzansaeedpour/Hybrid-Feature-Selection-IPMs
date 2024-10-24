


import numpy as np

def hypervolume(rp,pf):
    pf = pf[pf[:, 0].argsort()[::-1]]
    temp = np.array([])
    hv = np.array([])

    for i in range(pf.shape[0]):
        temp = np.array([])
        for j in range(len(rp)):
            if i==0:
                temp = np.append(temp, np.abs(rp[j]-pf[i,j]))
            else:
                if j==0:
                    temp = np.append(temp, np.abs(pf[i-1,j]-pf[i,j]))
                else:
                    temp = np.append(temp, np.abs(rp[j]-pf[i,j]))
        hv= np.append(hv, np.prod(temp))
    hv = np.sum(hv)
    return hv
