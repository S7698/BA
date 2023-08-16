import numpy as np
import matplotlib.pyplot as plt

def x_plot(file):
    data = np.loadtxt(file)
    pcol = [[1, 1, 1], [0, 0.75, 0], 'r']
    yltext = ['X', 'Y', 'Z']
    ttext = ['sensor ankle', 'sensor knee', 'sensor hip']
    
    #plt.clf() notwendig?
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    
    for sensorpos in range(3):
        for sensoraxis in range(3):
            ax = axs[sensoraxis, sensorpos]
            
            # Plot the patches: find the discontinuities in the labels
            f = np.where(data[1:, 10] - data[:-1, 10])[0]
            f = np.concatenate(([0], f, [data.shape[0] -1]))
            
            


            for i in range(len(f) - 1):
                x1 = data[f[i] + 1, 0] / 1000  # Time of start in ms 
                x2 = data[f[i + 1], 0] / 1000  # Time of end in ms
                type = data[f[i] + 1, 10]
                y1 = -3500
                y2 = -3000                
                
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=pcol[int(type)]))
            
            ax.plot(data[:, 0] / 1000, data[:, 1 + sensorpos * 3 + sensoraxis])
            
            ax.set_xlim(data[0, 0] / 1000, data[-1, 0] / 1000)
            ax.set_ylim(-3500, 3000)
            
            ax.set_xlabel('time [s]')
            ax.set_ylabel(f'Acc {yltext[sensoraxis]} [mg]')
            ax.set_title(ttext[sensorpos])
            
    plt.tight_layout()
    plt.show()
