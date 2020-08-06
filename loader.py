import numpy as np
import struct
from skimage import filters, morphology, io, color
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def load(fn, dtype = '<f4', roi = None, skip_n_bytes = 0):
    f = open(fn, 'rb')
    if skip_n_bytes > 0:
        _ = f.read(skip_n_bytes)

    shape = struct.unpack('iii', f.read(4*3))
    print ("shape of data: " + str(shape))
    N = shape[0] * shape[1] * shape[2]
    oct = np.frombuffer(f.read(), dtype=dtype)
    f.close()
    print ("loaded %i numbers of dtype %s" % (len(oct), dtype))

    oct = np.log( (oct * 1e41) + 0.00001 ) #avoid zero value

    bscan = oct.reshape(shape, order = 'F' )
    bscan = bscan.swapaxes(2, 1).swapaxes(1, 0) # swap for bscan on first axis
    bscan = np.array([b[::-1] for b in bscan])  # flip images
    return bscan

def preprocess(bscan, merge = 9, roi = None):
    median_selem = morphology.disk(2)[1:-1,:]
    if roi is not None:
        t,b, l,r = roi
        bscan = bscan[:,t:b,l:r]

    def _preprocess(i):
        fs, fe = i - merge//2, i+ merge//2 + 1
        fs = fs if fs >=0 else 0
        fe = fe if fe < len(bscan) else len(bscan)
        I = bscan[fs:fe].mean(axis=0)
        I = I - I.min()
        I = filters.median(I, selem = median_selem)
        I -= I.min() # normalize per frame
        I /= I.max()
        return I

    bscan_p = Parallel(n_jobs=10)(delayed(_preprocess)(i) for i in range(len(bscan)))
    bscan_p = np.array(bscan_p)

    return bscan_p

def load_mask(fn):
    bscan_mask = io.imread(fn)[:,:,:3]
    xx, yy, _ = bscan_mask.nonzero()
    segments_colors = np.unique(bscan_mask[xx, yy], axis=0)
    segments = []
    for sc in segments_colors:
        segm = (bscan_mask==sc).all(axis=2)
        segm = morphology.skeletonize(segm)
        segm = np.array(segm.nonzero())
        segm = segm[:,np.argsort(segm[1])]
        segments.append(segm)
    return segments, segments_colors

def plot_segments(I, segments, segments_colors, margin = True):
    if len(I.shape) == 2:
        I = color.gray2rgb(I)
    for segm, segm_color in zip(segments, segments_colors):
        I[segm[0], segm[1]] = (I[segm[0], segm[1]] + segm_color / 255 ) /2
        if margin:
            I[segm[0]-1, segm[1]] = (I[segm[0], segm[1]] + segm_color / 255 ) /2
            I[segm[0]+1, segm[1]] = (I[segm[0], segm[1]] + segm_color / 255 ) /2
    return I

if __name__ == "__main__":
    fn = './data/mouse_davis_OCT1_Amp.bin'
    bscan = load(fn)
    plt.figure(figsize=(10,10)); plt.imshow(bscan[-1,280:950]); plt.show()
    roi = (280, 950, 0, -1)
    bscan_p = preprocess(bscan, roi=roi)
    plt.figure(figsize=(10,10)); plt.imshow(bscan_p[0]); plt.show()
    plt.figure(figsize=(10,10)); plt.imshow(bscan_p[-1,280:950]); plt.show()
