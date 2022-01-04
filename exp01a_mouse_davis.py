import loader, dline
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, color

#%% 1. LOAD AND CHOOSE ROI

fn = './data/mouse_davis_OCT2_part4.bin'
bscan = loader.load(fn, dtype='f8')

ROI = (280, 950, 0, -1) #area of interest

bscan_p = loader.preprocess(bscan, roi=ROI, merge = 3)

plt.figure(figsize=(10,10));
plt.imshow(bscan_p[0]);
plt.show()

#%% 2. SAVE MASK PREPARE MASK


plt.figure(figsize=(10,10));
plt.imshow(bscan_p[0]);
plt.show()

io.imsave('to_mask.png', (bscan_p[0] * 255).astype(np.uint8))

#### OPEN to_mask.png IN PHOTOEDITOR AND DEINE TRACKING LINES

#%% 3. LOAD MASK

fn = './data/mouse_davis_OCT1_part3_layer0_mask_roi.png'
segments, segments_colors = loader.load_mask(fn)

I = loader.plot_segments(bscan_p[0].copy(), segments, segments_colors)
plt.figure(figsize=(20,10))
plt.imshow(I[200:450])
plt.show()

#%% (optional) SAVE BSCANS

for i in np.arange(0, len(bscan)):
    I = bscan_p[i]
    fig, (ax1) = plt.subplots(figsize=(20, 10), nrows=1)
    ax1.imshow(I[250:450], cmap='gray')
    plt.tight_layout()
    fig.savefig('./bscan/' + str(i).rjust(4,'0')+ '.png')
    plt.close(fig)

#%% 5. RUN DOUBLE LINE AND SAVE RESULTS

OUT_DIR = './out/'
segments_corrected = {0: segments}
import importlib
dline = importlib.reload(dline)

for i in np.arange(0, len(bscan)//2):
    I = bscan_p[i]
    key = np.argmin(np.abs(np.array(list(segments_corrected.keys())) - i))
    closest_idx = np.array(list(segments_corrected.keys()))[key]
    segm_corrected = []
    # print ('using ', closest_idx)
    for segm in segments_corrected[closest_idx]:
        segm_c, dists = dline.correct_line(segm, I, iters = 20000, debug_plot = False)
        segm_corrected.append(segm_c)

    segments_corrected[i] = segm_corrected

    ## this part is for saving results
    # Ic = loader.plot_segments(I, segm_corrected, segments_colors)
    # fig, (ax1, ax2) = plt.subplots(figsize=(20, 10), nrows=2)
    #
    # ax1.imshow(I[250:450], cmap='gray')
    # ax2.imshow(Ic[250:450])
    # plt.tight_layout()
    # fig.savefig(OUT_DIR + str(i).rjust(4,'0')+ '.png')
    # plt.close(fig)
    #
    # mask = np.zeros(I.shape)
    # mask = loader.plot_segments(mask, segm_corrected, segments_colors, margin=False)
    # mask /= mask.max()
    # mask = (mask*255).astype(np.uint8)
    # mask[(mask == [0,0,0]).all(axis=2)] = [255,255,255]
    # io.imsave(OUT_DIR + 'mask' + str(i).rjust(4,'0')+ '.png', mask)
