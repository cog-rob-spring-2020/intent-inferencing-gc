"""
16.412 Intent Inference GC | plotting.py
Plots results of running CVM. Adapted from plotting script received from
Christoph Scholler.
Author: Abbie Lee (abbielee@mit.edu)
"""
import numpy as np
import matplotlib.pyplot as plt
import gif
from tqdm import tqdm

im = plt.imread("background.jpg")

def gen_frame(detection_trajs):
    XI = 98.118385
    XF = 169.524979
    YI = 170.196945
    YF = 226.819290

    xlim = [XI, XF]
    ylim = [YF, YI]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    # set background
    ax.imshow(im, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])

    ax.tick_params(labelsize=14)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    for trajectory in detection_trajs:
        true = trajectory["true"][0]
        pred = trajectory["predicted"][0]
        hist = trajectory["observed"][0]
        ts = trajectory["ts"]

        # find out of bounds points and delete
        true_idxs = []
        for i in range(true.shape[0]):
            if true[i, 0] < xlim[0] or true[i, 0] > xlim[1] or \
               true[i, 1] < ylim[1] or true[i, 1] > ylim[0]:
                true_idxs.append(i)

        pred_idxs = []
        for i in range(pred.shape[0]):
            if pred[i, 0] < xlim[0] or pred[i, 0] > xlim[1] or \
               pred[i, 1] < ylim[1] or pred[i, 1] > ylim[0]:
                pred_idxs.append(i)

        hist_idxs = []
        for i in range(hist.shape[0]):
            if hist[i, 0] < xlim[0] or hist[i, 0] > xlim[1] or \
               hist[i, 1] < ylim[1] or hist[i, 1] > ylim[0]:
                hist_idxs.append(i)

        true = np.delete(true, true_idxs, axis=0)
        pred = np.delete(pred, pred_idxs, axis=0)
        hist = np.delete(hist, hist_idxs, axis=0)

        if 0 in true.shape:
            true = np.array([[0, 0]])

        if 0 in pred.shape:
            pred = np.array([[0, 0]])

        if 0 in hist.shape:
            hist = np.array([[0, 0]])

        # plot observed_history
        ax.plot(hist[:,0], hist[:,1], 'o', fillstyle="none", color='r', alpha=0.5, markersize=8)

        # plot gt
        ax.plot(true[:,0], true[:,1], 'o', fillstyle="none", color='g', alpha=0.5, markersize=8)

        # plot pred
        ax.plot(pred[:,0], pred[:,1], 'o', fillstyle="none", color='b', alpha=0.5, markersize=8)

        # plot current point
        ax.plot(true[0,0], true[0,1], 'o', color='k', alpha=1.0, markersize=10)

    # Make legend
    x, y = [], []
    l1, = ax.plot(x, y, 'o', fillstyle="none", color='r', alpha=0.5, markersize=8, label='history')
    l2, = ax.plot(x, y, 'o', color='k', markersize=10, label='current')
    l3, = ax.plot(x, y, 'o', fillstyle="none", color='g', alpha=0.5, markersize=8, label='true future')
    l4, = ax.plot(x, y, 'o', fillstyle="none", color='b', alpha=0.5, markersize=8, label='prediction')
    ax.legend((l1, l2, l3, l4), ("history", "current", "true future", "prediction"), loc="lower right")

    ax.set_title("t = " + str(int(ts)), fontsize=16)

@gif.frame
def gen_gif_img(ts):
    """
    ts: list of trajectories occuring in timestep ts
    """
    gen_frame(ts)

def plotting_gif(trajectories, outpath):
    """
    trajectories: list of length number of timesteps, where each timestep is a
                  list of trajectories in that timestep
    outpath: relative path to save gif
    """
    imgs = []
    for ts in tqdm(trajectories):
        imgs.append(gen_gif_img(ts))

    gif.save(imgs, outpath, duration=300)

def plotting_saveimgs(trajectories, outpath):
    counter = 0
    for ts in tqdm(trajectories):
        gen_frame(ts)
        plt.savefig(outpath + "/%06d.jpg" % counter)
        plt.close()
        counter += 1

def plotting(trajectories):
    gen_frame(trajectories)
    plt.show()
