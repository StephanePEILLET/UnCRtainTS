import numpy as np
from pathlib import Path 
from matplotlib import pyplot as plt
from os.path import basename
from itertools import product, repeat

IMG_NB = 1

inferences_dir = Path('/media/DATA/ADeWit/3STR/inference/utae/export/epoch_1/test')
targets_dir = [inferences_dir / 'img-{}_target_t-{}.npy'.format(str(IMG_NB), str(j)) for j in range(10)]

targets = [np.load(t) for t in targets_dir]
targets = [(targets[i]-np.min(targets[i]))/(np.max(targets[i])-np.min(targets[i]))*255 for i in range(len(targets))]
targets = [targets[i].astype(int) for i in range(len(targets))]
r = targets[0][np.newaxis, :]
v = targets[1][np.newaxis, :]
b = targets[2][np.newaxis, :]
rvb = np.concatenate((r,v,b), axis=0).swapaxes(0,2)

var_ = inferences_dir / 'img-{}_var.npy'.format(str(0))
var = np.load(var_).swapaxes(0,2)
for i in range(3):
    var[:,:,i] = (var[:,:,i]-np.min(var[:,:,i]))/(np.max(var[:,:,i])-np.min(var[:,:,i]))*255
var = var.astype(int)

pred_ = inferences_dir / 'img-{}_pred.npy'.format(str(0))
pred = np.load(pred_).swapaxes(0,2)
for i in range(pred.shape[2]):
    pred[:,:,i] = (pred[:,:,i]-np.min(pred[:,:,i]))/(np.max(pred[:,:,i])-np.min(pred[:,:,i]))*255
pred = pred.astype(int)

# =========================
# PLOT
# =========================
fig, axs = plt.subplots(5, 3, figsize=(10,10), dpi=300, layout='constrained')
#fig.set_size_inches(16, 12)
for ax, data in zip(axs.flat, targets):
    ax.imshow(data)
    ax.set_xticks([])
    ax.set_yticks([])
axs[0,0].set_xlabel("Bande B2")
axs[0,1].set_xlabel("Bande B3")
axs[0,2].set_xlabel("Bande B4")
axs[1,0].set_xlabel("Bande B8")
axs[1,1].set_xlabel("Bande B5")
axs[1,2].set_xlabel("Bande B6")
axs[2,0].set_xlabel("Bande B7")
axs[2,1].set_xlabel("Bande B8A")
axs[2,2].set_xlabel("Bande B11")
axs[3,0].set_xlabel("Bande B12")

axs[4,0].imshow(var, vmin=0, vmax=255)
axs[4,0].set_xlabel("Variance prediction")

axs[4, 1].imshow(pred[:,:,:3], vmin=0, vmax=255)
axs[4, 1].set_xlabel("Prediction B2/B3/B4")

axs[4, 2].imshow(rvb)
axs[4, 2].set_xlabel("Target B2/B3/B4")

fig.delaxes(axs[3,1])
fig.delaxes(axs[3,2])
fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
plt.show()
plt.savefig('/media/DATA/ADeWit/3STR/inference/utae/export/fig_' + str(IMG_NB) + '.png')