import os

datadir = '../dataset/'
SR = 64
stepSize = 32
offDelay = 2
onDelay = 2

TH.freeze = [3, 1.5, 3, 1.5, 1.5, 1.5, 3, 3, 1.5, 3]
TH.power = [2 ** 12]

for isubject in range(1, 4):
    for isensor in range(1):
        for iaxis in range(1):

            print(f'Subject {isubject:02d} sensor {isensor} axis {iaxis}')

            fileruns = [file for file in os.listdir(datadir) if file.startswith(f'S{isubject:02d}R')]
            resrun = [0, 0, 0, 0, 0]

            for filename in fileruns:
                print(f'\tProcessing {filename}')

                data = np.loadtxt(os.path.join(datadir, filename))

                res = x_fi(data[:, 2 + isensor * 3 + iaxis], SR, stepSize)

                res.quot[res.sum < TH.power] = 0

                lframe = res.quot > TH.freeze[isubject]

                gtframe = data[res.time, 10]
                xp = np.where(gtframe != 0)[0]

                gtframe2 = gtframe[xp] - 1
                lframe2 = lframe[xp]

                res = x_countTxFx(gtframe2, lframe2, offDelay * SR / stepSize, onDelay * SR / stepSize)
                resrun = resrun + res

                print(f'\t\tAxis {iaxis}. TP: {res[0]}  TN: {res[1]} FP: {res[2]} FN: {res[3]}. Tot freeze: {res[4]}')

            print(f'\tTotal TP: {resrun[0]}  TN: {resrun[1]} FP: {resrun[2]} FN: {resrun[3]}. Tot freeze: {resrun[4]}')
            print(f'\tSensitivity: {resrun[0] / (resrun[0] + resrun[3]):.2f} Specificity: {resrun[1] / (resrun[1] + resrun[2]):.2f}')

