import os
import numpy as np
import matplotlib.pyplot as plt
from delay_plot import delay_plotting
from plotsBA import x_plot
from xfiBA import x_fi
from xcountBA import x_countTxFx
from comparelistsBA import compare_lists

datadir = '../dataset/'
SR = 64
stepSize = 32
offDelay = 2 #offDelay/onDelay: tolerance for latency in the algorithm
onDelay = 2
freeze = [3, 1.5, 3, 1.5, 1.5, 1.5, 3, 3, 1.5, 3] # individual threshold for every participant
TH_power = 2 ** 12.0


for isubject in range(2, 3):
    for isensor in range(1):
        for iaxis in range(1, 2):

            print(f'Subject {isubject:02d} sensor {isensor} axis {iaxis}')

            fileruns = [file for file in os.listdir(datadir) if file.startswith(f'S{isubject:02d}R')]
            resrun = [0, 0, 0, 0, 0]

            for filename in fileruns:
                print(f'\tProcessing {filename}')

                data = np.loadtxt(os.path.join(datadir, filename))


                # Moore's Algorithm
                res = x_fi(data[:, 1 + isensor * 3 + iaxis], SR, stepSize)


                res['quot'] = np.array(res['quot'])
                res['sum'] = np.array(res['sum'])
                res['quot'][res['sum'] < TH_power] = 0
                freeze_isubject = freeze[isubject-1]

                lframe = []
                freeze_index = []

                # lframe as boolean list 
                for quo in res['quot']:
                    a = float(quo)
                    lframe.append(a > freeze_isubject)
                    freeze_index.append(a)

                lframe =np.array(lframe)
                freeze_index = np.array(freeze_index)

                # only experiment parts used
                gtframe = data[res['time'], 10]
                xp = np.where(gtframe != 0)[0]
                gtframe2 = gtframe[xp] - 1
                # take as function
                lframe2 = lframe[xp]
                freeze_index = freeze_index[xp]

                # Call the function with the lists
                delay = compare_lists(lframe2, gtframe2, freeze_index, freeze_isubject)
                #delay_plotting(lframe2, gtframe2, freeze_index, freeze_isubject)
                res = x_countTxFx(gtframe2, lframe2, offDelay * SR / stepSize, onDelay * SR / stepSize)
                resrun = [resrun[i] + res[i] for i in range(len(resrun))]

                print(f'\t\tAxis {iaxis}. TP: {res[0]}  TN: {res[1]} FP: {res[2]} FN: {res[3]}. Tot freeze: {res[4]}')
                print(f'\t\tDetection Delay: FP: {delay[0]}  FN: {delay[1]} Total Delay: {delay[2]} STD: {delay[3]}.')
                #x_plot(filename)
            
                
            print(f'\tTotal TP: {resrun[0]}  TN: {resrun[1]} FP: {resrun[2]} FN: {resrun[3]}. Tot freeze: {resrun[4]}')
            print(f'\tSensitivity: {resrun[0] / (resrun[0] + resrun[3]):.2f} Specificity: {resrun[1] / (resrun[1] + resrun[2]):.2f}')

