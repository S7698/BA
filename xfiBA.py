import numpy as np

# Compute the freezing index
# SR: Sample rate in herz
# Original version allowed various FFT sizes - now FFT and window size must be equal

def x_fi(data, SR, stepSize):
    NFFT = 256
    locoBand = [0.5, 3]
    freezeBand = [3, 8]
    windowLength = 256

    f_res = SR / NFFT
    f_nr_LBs = round(locoBand[0] / f_res)
    f_nr_LBs = np.array([f for f in f_nr_LBs if f != 0])
    f_nr_LBe = round(locoBand[1] / f_res)
    f_nr_FBs = round(freezeBand[0] / f_res)
    f_nr_FBe = round(freezeBand[1] / f_res)

    d = NFFT / 2

    jPos = windowLength + 1
    i = 1
    time = []
    sumLocoFreeze = []
    freezeIndex = []

    while jPos <= len(data):
        jStart = jPos - windowLength + 1
        time.append(jPos)

        y = data[jStart:jPos]
        y = y - np.mean(y)

        Y = np.fft.fft(y, NFFT)
        Pyy = Y * np.conj(Y) / NFFT

        areaLocoBand = np.trapz(Pyy[f_nr_LBs:f_nr_LBe], dx=1 / SR)
        areaFreezeBand = np.trapz(Pyy[f_nr_FBs:f_nr_FBe], dx=1 / SR)

        sumLocoFreeze.append(areaFreezeBand + areaLocoBand)
        freezeIndex.append(areaFreezeBand / areaLocoBand)

        jPos = jPos + stepSize
        i = i + 1

    res = {
        'sum': sumLocoFreeze,
        'quot': freezeIndex,
        'time': time
    }

    return res

