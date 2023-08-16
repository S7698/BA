import numpy as np


def x_numericalIntegration(x, SR):
    i = (np.sum(x[1:]) / SR + np.sum(x[:-1]) / SR) / 2
    return i

# Compute the freezing index
# SR: Sample rate in herz
# Original version allowed various FFT sizes - now FFT and window size must be equal

def x_fi(data, SR, stepSize):
    NFFT = 256 #number of points used for the Fast Fourier Transform
    locoBand = [0.5, 3] #frequency range for the local motion band
    freezeBand = [3, 8] #frequency range for the local freeze band
    windowLength = 256

    f_res = SR / NFFT #frequency resolution
    # local/freeze band start/end
    f_nr_LBs = np.array([int(locoBand[0] / f_res)])
    f_nr_LBs = f_nr_LBs[f_nr_LBs != 0]
    f_nr_LBe = np.array([int(locoBand[1] / f_res)])
    f_nr_FBs = int((freezeBand[0] / f_res))
    f_nr_FBe = int((freezeBand[1] / f_res))

    d = NFFT / 2

    jPos = windowLength + 1
    i = 1
    time = []
    sumLocoFreeze = []
    freezeIndex = []
    amount = 0

    # Calculation of the Freeze Index
    while jPos <= len(data):
        jStart = jPos - windowLength + 1
        time.append(jPos)
        y = data[jStart-1:jPos]
        amount = jPos-jStart-1
        y = y - np.mean(y)
        Y = np.fft.fft(y, NFFT)
        Pyy = Y * np.conj(Y) / NFFT
        a1 = int(f_nr_LBs)-1
        a2 = int(f_nr_LBe)
        b = int(f_nr_FBs)-1
        areaLocoBand = x_numericalIntegration(Pyy[a1:a2], SR)
        areaFreezeBand = x_numericalIntegration(Pyy[b:f_nr_FBe], SR)
        sumLocoFreeze.append(areaFreezeBand + areaLocoBand)
        freezeIndex.append(areaFreezeBand / areaLocoBand)
        jPos = jPos + stepSize
        #with open("Freeze_P.txt", 'a') as file:
        #   file.write(str(float(areaFreezeBand / areaLocoBand)) + '\n')

        i = i + 1

    res = {
        'sum': sumLocoFreeze,
        'quot': freezeIndex,
        'time': time
    }
    
    return res

