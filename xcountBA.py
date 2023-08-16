import numpy as np

#Count the true pos, false pos, etc in frames
#gtframe: column vector of ground truth of frame
#lframe: column vector classification result of frame
#offDelay/onDelay: tolerance for latency in the algorithm. Specified in frames.

#The latency-tolerance can only be used with binary labels : 0=nothing, 1=event
#Returns: [TP TN FP FN Nev]
# Nev: number of events in the ground truth data
 
    

def x_countTxFx(gtframe, lframe, offDelay, onDelay):
    # Returns: [TP TN FP FN Nev]
    # Nev: number of events in the ground truth data

    # Want here to create labels tolerating algorithm latency in the
    # transitions from nothing->event and event->nothing. 
    # For this we need gtframedelayoff and grframedelayon that are 
    # variants of gtframe with delay.
    # This is built using a help 'labels' array.
    
    

    # Convert the frame labels to the format: [fromsample tosample]
    # finds transitions from 0 to 1 or from 1 to 0 and returns these positions
    f = (np.where(gtframe[1:] - gtframe[:-1])[0]) +1
    f = np.concatenate(([0], f, [len(gtframe)]))
    
    # Convert to labels [fromframe toframe] where there is an event
    labels = []
    for li in range(len(f) - 1):
        if gtframe[f[li]] == 1:
            labels.append([f[li] +1, f[li + 1]])

    gtframedelayoff = np.zeros_like(gtframe)
    gtframedelayon = np.zeros_like(gtframe)
    s = np.arange(1, len(gtframe) + 1)
    for li in range(len(labels)):
        s_index = (np.where(s >= labels[li][0])[0][0])+1
        e_index = (np.where(s <= labels[li][1])[0][-1])+1
        e_indexOff = np.where(s <= labels[li][1] + offDelay)[0][-1]
        gtframedelayoff[s_index-1:e_indexOff+1] = 1
        s_indexOn = np.where(s >= labels[li][0] + onDelay)[0][0]
        gtframedelayon[s_indexOn:e_index] = 1
    res_vec = np.zeros((len(gtframe), 6))  # TP TPd TN TNd FP FN

    i_TX = np.where(gtframe == lframe)[0]
    i_TP = np.where(lframe[i_TX] == 1)[0]
    res_vec[i_TX[i_TP], 0] = 1
    i_TN = np.where(lframe[i_TX] == 0)[0]
    res_vec[i_TX[i_TN], 2] = 1

    #mark all false detected (FP) and missed (FN) time-slots
    i_FX = np.where(gtframe != lframe)[0]
    i_FP = np.where(lframe[i_FX] == 1)[0]
    res_vec[i_FX[i_FP], 4] = 1
    i_FN = np.where(lframe[i_FX] == 0)[0]
    res_vec[i_FX[i_FN], 5] = 1

    # compare with delay tolerance
    # TPd : time-slots true due to the off delay
    i_X = np.where(res_vec[:, 4] == gtframedelayoff)[0] #initial assesment says FP, but delay tolerance is negative -> change to TP
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
