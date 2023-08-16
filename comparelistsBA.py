import os
import numpy as np
import matplotlib.pyplot as plt

def compare_lists(lframe2, gtframe2, index, freeze_isubject):
    # Returns: [FP, FN, Total Delay, STD]
    # Comparsion of events and detected events, calculate the delay in detection

    # Get start and end point of detected events and labelled events 
    f1 = (np.where(gtframe2[1:] - gtframe2[:-1])[0]) +1
    f1 = np.concatenate(([0], f1, [len(gtframe2)]))
    f2 = (np.where(lframe2[1:] != lframe2[:-1])[0]) +1
    f2 = np.concatenate(([0], f2, [len(lframe2)]))
    gtlabels = []
    for li in range(len(f1) - 1):
        if gtframe2[f1[li]] == 1:
            gtlabels.append([f1[li] + 1, f1[li + 1]])
    llabels = []
    for li in range(len(f2) - 1):
        if lframe2[f2[li]] == 1:
            llabels.append([f2[li] + 1, f2[li + 1]])
   

    # Compare detection and events and find delay 
    ergebnisse = []
    delay = []
    fn = []
    detected_events = []
    true_detections = []
    fp = []
    if len(llabels) > 0 and len(gtlabels) > 0:
        for detection in llabels:
            dstart = detection[0]
            dende = detection[1]
            for event in gtlabels:
                estart = event[0] 
                eende = event[1]              
                if dstart <= eende and dende >= estart and estart-dstart <= 10:
                    if ergebnisse == [] or (estart != ergebnisse[-1][1] and dstart != ergebnisse[-1][0]):
                        ergebnisse.append([dstart, estart])  
                        detected_events.append(estart)
                        true_detections.append(dstart)
                        delay.append(abs(dstart- estart))
                    elif ergebnisse == [] or dstart != ergebnisse[-1][0]:
                        true_detections.append(dstart)
                    
    else:
        print("One or both of the lists are empty.")    
    
    # FN
    for event in gtlabels:
        x = event[0]
        if x not in detected_events:
            fn.append(event)
    # FP
    for event in llabels:
        x = event[0]
        if x not in true_detections:
            fp.append(event)
    
    total_delay = sum(delay)        
    std = np.std(delay)

    # Positive and Negative Predictive Value
    ppv = len(true_detections)/(len(true_detections)+len(fp))
    npv = len(detected_events)/(len(detected_events)+len(fn))
    print("PPV: ", ppv, "NPV: ", npv)

    res = [len(fp), len(fn), total_delay, std]
    return res
