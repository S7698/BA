import numpy as np

#Count the true pos, false pos, etc in frames
#gtframe: column vector of ground truth of frame
#lframe: column vector classification result of frame
#offDelay/onDelay: tolerance for latency in the algorithm. Specified in frames.

#The latency-tolerance can only be used with binary labels : 0=nothing, 1=event
#Returns: [TP TN FP FN Nev]
# Nev: number of events in the ground truth data
 
    

def x_countTxFx(gtframe, lframe, offDelay, onDelay):
   
    #Want here to create labels tolerating algorithm latency in the
    # transitions from nothing->event and event->nothing. 
    # For this we need gtframedelayoff and grframedelayon that are 
    # variants of gtframe with delay.
    # This is built using a help 'labels' array.
    
    f = np.where(np.diff(gtframe) != 0)[0]
    f = np.concatenate(([0], f, [len(gtframe)]))
    labels = []
    for li in range(1, len(f)):
        if gtframe[f[li]] == 1:
            labels.append([f[li], f[li + 1]])

    gtframedelayoff = np.zeros_like(gtframe)
    gtframedelayon = np.zeros_like(gtframe)
    s = np.arange(1, len(gtframe) + 1)
    for li in range(len(labels)):
        s_index = np.where(s >= labels[li][0])[0][0]
        e_index = np.where(s <= labels[li][1])[0][-1]
        e_indexOff = np.where(s <= labels[li][1] + offDelay)[0][-1]
        gtframedelayoff[s_index:e_indexOff + 1] = 1
        s_indexOn = np.where(s >= labels[li][0] + onDelay)[0][0]
        gtframedelayon[s_indexOn:e_index + 1] = 1

    res_vec = np.zeros((len(gtframe), 6))  # TP TPd TN TNd FP FN

    i_TX = np.where(gtframe == lframe)[0]
    i_TP = np.where(lframe[i_TX] == 1)[0]
    res_vec[i_TX[i_TP], 0] = 1
    i_TN = np.where(lframe[i_TX] == 0)[0]
    res_vec[i_TX[i_TN], 2] = 1

    i_FX = np.where(gtframe != lframe)[0]
    i_FP = np.where(lframe[i_FX] == 1)[0]
    res_vec[i_FX[i_FP], 4] = 1
    i_FN = np.where(lframe[i_FX] == 0)[0]
    res_vec[i_FX[i_FN], 5] = 1

    i_X = np.where(res_vec[:, 4] == gtframedelayoff)[0]
    i_TPd = np.where(res_vec[i_X, 4] == 1)[0]
    res_vec[i_X[i_TPd], 1] = 1
    res_vec[i_X[i_TPd], 4] = 0

    i_X = np.where(res_vec[:, 5] != gtframedelayon)[0]
    i_TNd = np.where(res_vec[i_X, 5] == 1)[0]
    res_vec[i_X[i_TNd], 3] = 1
    res_vec[i_X[i_TNd], 5] = 0

    TP = np.sum(res_vec[:, 0]) + np.sum(res_vec[:, 1])
    TN = np.sum(res_vec[:, 2]) + np.sum(res_vec[:, 3])
    FP = np.sum(res_vec[:, 4])
    FN = np.sum(res_vec[:, 5])

    res = [TP, TN, FP, FN, len(labels)]
    return res

