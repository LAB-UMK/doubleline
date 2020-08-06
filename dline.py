import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numba

def dline_cost(I, l, d):
    m  = (I[l[0], l[1] ]) # to ma byc wysokie
    d1 = (I[l[0]-d, l[1] ]) # a to niskie
    d2 = (I[l[0]+d, l[1] ]) # a to niskie
    return (m - d1).sum() + (m - d2).sum()

def smooth_line(l):
    return np.array([savgol_filter(l[0], 101, 2) , l[1]])

def correct_line(seed_line, I, iters = 30000, debug_plot = False, eps = 1.):
    l = seed_line.copy()
    dists, dists_on_filter = [], []
    for iter in range(iters):
        lm = l.copy()
        lm[:, np.random.randint(0, lm.shape[1])] += [np.random.choice([-1, 1]), 0]
        a = dline_cost(I, l , 5)
        b = dline_cost(I, lm, 5)

        if a < b: l = lm

        dists.append((a))

        if iter % 500 == 0:
            ls = np.round(smooth_line(l)).astype(int)
            l = ls.copy()
            dist_f = dline_cost(I, l , 5)
            if len(dists_on_filter) > 0:
                if (np.abs(dists_on_filter[-1] - dist_f) < eps):
                    break

            dists_on_filter.append(dist_f)

            if debug_plot:
                plt.figure(figsize=(20,10))
                plt.imshow(I, cmap ='gray')
                plt.scatter(seed_line[1], seed_line[0], c='r', s=1, alpha = 0.8)
                plt.scatter(l[1], l[0], c='g', s=1, alpha = 0.8)
                plt.scatter(ls[1], ls[0], c='b', s=1, alpha = 0.8)

                plt.show()

    return ls, dists
