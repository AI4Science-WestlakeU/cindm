import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

for simnum in range(999,1000): 
    vel = True
    path = "/home/ubuntu/lily-pad/LilyPad/dataset/naca_ellipse/training_trajectories/sim_{:06d}/".format(simnum)
    savepath="/home/ubuntu/lily-pad/LilyPad/visualization/"
    if vel:
        pdf = PdfPages(savepath+"simvel_{:06d}.pdf".format(simnum))
    else:
        pdf = PdfPages(savepath+"simprs_{:06d}.pdf".format(simnum))

    for i in range(0,200):
        if i%3==0:
            if vel:
                feature = np.load(path+"velocity_{:06d}.npy".format(i))
                fig, ax = plt.subplots(figsize=(8,4),ncols=2)
                mappable0 = ax[0].imshow(feature[0,:,:], cmap='viridis',
                                         #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                         aspect='auto',
                                         origin='lower')
                mappable1 = ax[1].imshow(feature[1,:,:], cmap='viridis',
                                         #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                         #interpolation="bicubic",
                                         aspect='auto',
                                         origin='lower')
                fig.colorbar(mappable0, ax=ax[0])
                fig.colorbar(mappable1, ax=ax[1])
                fig.tight_layout()
            else:
                feature = np.load(path+"pressure_{:06d}.npy".format(i))
                fig, ax = plt.subplots(figsize=(8,4),ncols=2)
                mappable0 = ax[0].imshow(feature[:,:], cmap='viridis',
                                         #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                         aspect='auto',
                                         origin='lower')
                mappable1 = ax[1].imshow(feature[:,:], cmap='viridis',
                                         #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                         #interpolation="bicubic",
                                         aspect='auto',
                                         origin='lower')
                fig.colorbar(mappable0, ax=ax[0])
                fig.colorbar(mappable1, ax=ax[1])
                fig.tight_layout()
            pdf.savefig()

    pdf.close()
