import loader, dline
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, color

#%% LOAD

fn = './data/mouse_davis_OCT1_Amp.bin'
ROI = (280, 950, 0, -1) #area of interest

bscan = loader.load(fn)
plt.figure(figsize=(10,10)); plt.imshow(bscan[-1,280:950]); plt.show()
bscan_p = loader.preprocess(bscan, roi=ROI)
plt.figure(figsize=(10,10)); plt.imshow(bscan[0]); plt.show()
plt.figure(figsize=(10,10)); plt.imshow(bscan_p[0]); plt.show()

#%% LOAD MASK

fn = './data/mouse_davis_OCT1_Amp_layer150_mask_roi.png'
segments, segments_colors = loader.load_mask(fn)

I = loader.plot_segments(bscan_p[151].copy(), segments, segments_colors)
plt.figure(figsize=(20,10))
plt.imshow(I[200:400])
plt.show()

#%%

import importlib
_ = importlib.reload(dline)


segments_corrected = {150: segments }


# for i in np.arange(150, len(bscan)):
for i in np.arange(150, 0, -1):
    I = bscan_p[i]
    key = np.argmin(np.abs(np.array(list(segments_corrected.keys())) - i))
    closest_idx = np.array(list(segments_corrected.keys()))[key]
    segm_corrected = []
    print ('using ', closest_idx)
    for segm in segments_corrected[closest_idx]:
        segm_c, dists = dline.correct_line(segm, I, iters = 20000, debug_plot = False)
        # plt.plot(dists)
        segm_corrected.append(segm_c)
    segments_corrected[i] = segm_corrected
    # plt.title(str(i))
    # plt.show()

    Ic = loader.plot_segments(I, segm_corrected, segments_colors)
    fig, (ax1, ax2) = plt.subplots(figsize=(20, 10), nrows=2)
    ax1.imshow(I[250:400], cmap='gray')
    ax2.imshow(Ic[250:400])
    plt.tight_layout()
    fig.savefig('./out/' + str(i).rjust(4,'0')+ '.png')
    plt.close(fig)

    mask = np.zeros(I.shape)
    mask = loader.plot_segments(mask, segm_corrected, segments_colors, margin=False)
    mask /= mask.max()
    mask = (mask*255).astype(np.uint8)
    mask[(mask == [0,0,0]).all(axis=2)] = [255,255,255]
    io.imsave('./out/mask' + str(i).rjust(4,'0')+ '.png', mask)

#%%
plt.figure(figsize=(10,20))
plt.imshow(bscan_p[150])
plt.show()
plt.figure(figsize=(10,20))
plt.imshow(bscan[150])
plt.show()
