import matplotlib.pyplot as plt
import numpy as np

def delay_plotting(lframe2, gtframe2, index, freeze_isubject):
    
    f1 = (np.where(gtframe2[1:] - gtframe2[:-1])[0]) +1
    f1 = np.concatenate(([0], f1, [len(gtframe2)]))
    gtlabels = []
    for li in range(len(f1) - 1):
        if gtframe2[f1[li]] == 1:
            gtlabels.append([f1[li] + 1, f1[li + 1]])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.bar(np.arange(1, 1+len(gtframe2)), gtframe2, width = 1.0, alpha = 0.7, label = 'Events')
    ax1.set_ylabel('gtframe2\nEvents')
    ax1.set_xlim(300, 500)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.vlines([i[0]-0.5 for i in gtlabels], 0, 1, color = 'r', label = 'Eventstart')
    ax1.legend(framealpha = 1.0, fontsize = 'small')

    ax2.bar(np.arange(1, 1+len(lframe2)), lframe2, width = 1.0, alpha = 0.7, label = 'Detections')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('lframe2\nDetection')
    ax2.set_xlim(300, 500)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.vlines([i[0]-0.5 for i in gtlabels], 0, 1, color = 'r', label = 'Eventstart')
    ax2.legend(framealpha = 1.0, fontsize = 'small')

    ax3.plot(index)
    ax3.set_xlim(300, 500)
    ax3.set_ylim(bottom = 0, top = 7)
    ax3.hlines(freeze_isubject, 300, 500, color = 'g', label = 'Freeze Detection Threshold')
    ax3.fill_between(np.arange(len(index)), freeze_isubject, index, where=index > freeze_isubject, alpha = 0.5)
    ax3.set_ylabel("Freezeindex")
    ax3.legend(framealpha = 1.0, fontsize = 'small', loc = 'upper left')
    plt.subplots_adjust(hspace=0)
    fig.suptitle("Delay in Event Detection")
    plt.show()