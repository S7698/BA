import os
import numpy as np
import matplotlib.pyplot as plt

datadir = '../dataset/'
SR = 64
stepSize = 32
offDelay = 2
onDelay = 2

#TH.freeze = [3, 1.5, 3, 1.5, 1.5, 1.5, 3, 3, 1.5, 3]
#TH.power = [2 ** 12]
freeze = [3, 1.5, 3, 1.5, 1.5, 1.5, 3, 3, 1.5, 3]
power = [2 ** 12]


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
    
    

    # Convert the frame labels to the format: [fromsample tosample]
    f = (np.where(gtframe[1:] - gtframe[:-1])[0]) +1
    f = np.concatenate(([0], f, [len(gtframe) - 1]))

    # Convert to labels [fromframe toframe] where there is an event
    labels = []
    for li in range(len(f) - 1):
        if gtframe[f[li] + 1] == 1:
            labels.append([f[li] + 1, f[li + 1]])



    gtframedelayoff = np.zeros_like(gtframe)
    gtframedelayon = np.zeros_like(gtframe)
    s = np.arange(1, len(gtframe) + 1)
    for li in range(len(labels)):
        s_index = (np.where(s >= labels[li][0])[0][0])+1
        e_index = (np.where(s <= labels[li][1])[0][-1])+1
        e_indexOff = np.where(s <= labels[li][1] + offDelay)[0][-1]
        gtframedelayoff[s_index:e_indexOff + 2] = 1
        s_indexOn = np.where(s >= labels[li][0] + onDelay)[0][0]
        gtframedelayon[s_indexOn:e_index + 2] = 1

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

    #print(np.sum(res_vec[:, 0]), np.sum(res_vec[:, 1]), np.sum(res_vec[:, 2]), np.sum(res_vec[:, 3]), np.sum(res_vec[:, 4]), np.sum(res_vec[:, 5]))
    TP = np.sum(res_vec[:, 0]) + np.sum(res_vec[:, 1])
    TN = np.sum(res_vec[:, 2]) + np.sum(res_vec[:, 3])
    FP = np.sum(res_vec[:, 4])
    FN = np.sum(res_vec[:, 5])
    
    res = [TP, TN, FP, FN, len(labels)]
    return res

def x_numericalIntegration(x, SR):
    i = (np.sum(x[1:]) / SR + np.sum(x[:-1]) / SR) / 2
    return i


def x_fi(data, SR, stepSize):
    NFFT = 256 #number of points used for the Fast Fourier Transform
    locoBand = [0.5, 3] #frequency range for the local motion band
    freezeBand = [3, 8]
    windowLength = 256

    f_res = SR / NFFT #frequency resolution
    #local/freeze band start/end
    f_nr_LBs = np.array([int(locoBand[0] / f_res)])
    f_nr_LBs = f_nr_LBs[f_nr_LBs != 0]
    f_nr_LBe = np.array([int(locoBand[1] / f_res)])
    f_nr_FBs = int((freezeBand[0] / f_res))
    f_nr_FBe = int((freezeBand[1] / f_res))

    d = NFFT / 2

    jPos = windowLength + 1
    i = 1
    #empty lists
    time = []
    sumLocoFreeze = []
    freezeIndex = []

    while jPos <= len(data):
        jStart = jPos - windowLength + 1
        time.append(jPos)

        y = data[jStart-1:jPos-1]
        y = y - np.mean(y)
        Y = np.fft.fft(y, NFFT)
        Pyy = Y * np.conj(Y) / NFFT
        a = int(f_nr_LBs)
        b = int(f_nr_LBe)
        areaLocoBand = x_numericalIntegration(Pyy[a:b], SR)
        areaFreezeBand = x_numericalIntegration(Pyy[f_nr_FBs:f_nr_FBe], SR)

        sumLocoFreeze.append(areaFreezeBand + areaLocoBand)
        freezeIndex.append(areaFreezeBand / areaLocoBand)

        jPos = jPos + stepSize
        i = i + 1
    #print(sumLocoFreeze)
    res = {
        'sum': sumLocoFreeze,
        'quot': freezeIndex,
        'time': time
    }
    
    return res

for isubject in range(2, 3):
    for isensor in range(1):
        for iaxis in range(1, 2):

            print(f'Subject {isubject:02d} sensor {isensor} axis {iaxis}')

            fileruns = [file for file in os.listdir(datadir) if file.startswith(f'S{isubject:02d}R')]
            resrun = [0, 0, 0, 0, 0]

            for filename in fileruns:
                print(f'\tProcessing {filename}')

                data = np.loadtxt(os.path.join(datadir, filename))

                res = x_fi(data[:, 1 + isensor * 3 + iaxis], SR, stepSize)
                print(len(res['sum']))
                plt.plot(res['sum'])
                plt.show()

                res['quot'][res['sum'] < power] = 0

                freeze_isubject = freeze[isubject-1]

                lframe = []

                for quo in res['quot']:
                    a = float(quo)
                    lframe.append(a > freeze_isubject)

                lframe =np.array(lframe)
                #lframe = [quote > freeze_isubject for quote in res['quot']]
                #lframe = (res['quot'] > freeze_isubject)

                # only experiment parts used
                gtframe = data[res['time'], 10]
                xp = np.where(gtframe != 0)[0]
                gtframe2 = gtframe[xp] - 1
                # take as function
                #lframe2 = np.array(lframe)[xp]
                lframe2 = lframe[xp]
                print(len(gtframe))
                p = np.where(lframe != 0)[0]
                res = x_countTxFx(gtframe2, lframe2, offDelay * SR / stepSize, onDelay * SR / stepSize)
                resrun = [resrun[i] + res[i] for i in range(len(resrun))]
                print(f'\t\tAxis {iaxis}. TP: {res[0]}  TN: {res[1]} FP: {res[2]} FN: {res[3]}. Tot freeze: {res[4]}')
            """
            matlab = []
            with open('index.txt', 'r') as file:
                for line in file:
                    print(line)
                    #matlab.append(float(line.strip()))
           """
                
            print(f'\tTotal TP: {resrun[0]}  TN: {resrun[1]} FP: {resrun[2]} FN: {resrun[3]}. Tot freeze: {resrun[4]}')
            print(f'\tSensitivity: {resrun[0] / (resrun[0] + resrun[3]):.2f} Specificity: {resrun[1] / (resrun[1] + resrun[2]):.2f}')

