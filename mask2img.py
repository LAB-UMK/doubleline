import numpy as np
from skimage import io
from glob import glob
import matplotlib.pyplot as plt
from scipy import io
%matplotlib

fns = glob('./out/mask*.png')
I = io.imread(fns[0])

bscans_NO = len(fns)
ascans_NO = I.shape[1]

O = np.zeros((bscans_NO, ascans_NO), dtype = int) *np.nan
O.shape


markers = np.unique(I.reshape((I.shape[0] * I.shape[1], 3)), axis=0)

m = markers[0] # tutaj wybierz ktory marker
for i, fn in enumerate(fns):
    print (fn)
    I = io.imread(fn)

    Im = (I == m).all(axis=2)
    line_coords = np.argwhere(Im)
    ln_Y, ln_X = line_coords.T

    O[i,ln_X] = ln_Y


# tutaj mozna zapisac macierz O
plt.imshow(O); plt.colorbar()
io.savemat('./out/O.mat', {'O': O})
