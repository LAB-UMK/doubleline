import loader, dline
import matplotlib.pyplot as plt
import numpy as np
from skimage import registration, filters, io
# importlib.reload(loader)

in_fn = './eye_data/2021-ORG/16_15_57_fbg_raw_blog_amp.bin'

bscan = loader.load(in_fn)
bscan_p = loader.preprocess(bscan, merge=1)

# bscan_f = np.zeros(bscan.shape)
#
# for i, F in enumerate(bscan):
#     bscan_f[i] = filters.gaussian(F, sigma=2)

#%%

registration_roi = (0, -1)
F0 = bscan[0][registration_roi[0]:registration_roi[1]]
pos = np.zeros((len(bscan), 2))
for i, F in enumerate(bscan_p):
    if i % 100 == 0 : print (i,'/', len(bscan))
    F = F[registration_roi[0]: registration_roi[1]]
    r, _, _  = registration.phase_cross_correlation(F0, F)
    pos[i] = r[::-1]



#%%

fig, (a1, a2) = plt.subplots(ncols=2)
R, a = loader.rever(bscan[::20], pos[::20])
a1.imshow(R)
a2.imshow(bscan[100])

#%%

fn = './eye_data/2021-ORG/to_mask_16_15_57.png'
segments, segments_colors = loader.load_mask(fn)

I = loader.plot_segments(bscan_p[151].copy(), segments, segments_colors)
plt.figure(figsize=(20,10))
plt.imshow(I[200:400])
plt.show()

#%%

ch = pos.T[1][1:] - pos.T[1][:-1]
ch = np.insert(ch, 0, 0).astype(int)
%matplotlib inline
segments_corrected = {0: segments }

i = 0
for i in np.arange(0, len(bscan)):
    I = bscan_p[i]
    key = np.argmin(np.abs(np.array(list(segments_corrected.keys())) - i))
    closest_idx = np.array(list(segments_corrected.keys()))[key]
    segm_corrected = []
    print ('using ', closest_idx)
    for segm in segments_corrected[closest_idx]:
        segm_f = segm.copy()
        segm_f[0] -= ch[i]
        segm_c, dists = dline.correct_line(segm_f, I, iters = 20000, debug_plot = False)
        # plt.plot(dists)
        segm_corrected.append(segm_c)
    segments_corrected[i] = segm_corrected
    # plt.title(str(i))
    # plt.show()

    Ic = loader.plot_segments(I, segm_corrected, segments_colors)

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 20), ncols=2)
    ax1.imshow(I[:], cmap='gray')
    ax2.imshow(Ic[:] / Ic.max())
    plt.tight_layout()
    fig.savefig('./out/' + str(i).rjust(4,'0')+ '.png')
    plt.close(fig)

    mask = np.zeros(I.shape)
    mask = loader.plot_segments(mask, segm_corrected, segments_colors, margin=False)
    mask /= mask.max()
    mask = (mask*255).astype(np.uint8)
    mask[(mask == [0,0,0]).all(axis=2)] = [255,255,255]
    io.imsave('./out/mask' + str(i).rjust(4,'0')+ '.png', mask)
