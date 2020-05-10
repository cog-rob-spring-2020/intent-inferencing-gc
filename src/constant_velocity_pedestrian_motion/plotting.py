"""
16.412 Intent Inference GC | plotting.py
Plots results of running CVM. Adapted from plotting script received from
Christoph Scholler.
Author: Abbie Lee (abbielee@mit.edu)
"""
import numpy as np
import matplotlib.pyplot as plt
import gif

im = plt.imread("CARLA.jpg")

def gen_frame(detection_trajs):

    xlim = [99, 171]
    ylim = [-230, -170]

    fig, ax = plt.subplots(figsize=(14, 12))
    # set background
    y_offset = 4
    ax.imshow(im, extent=[xlim[0], xlim[1], ylim[0]+y_offset, ylim[1]+y_offset])

    ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0]+y_offset, ylim[1]+y_offset)

    for trajectory in detection_trajs:
        true = trajectory["true"][0]
        pred = trajectory["predicted"][0]
        hist = trajectory["observed"][0]
        ts = trajectory["ts"]

        # mirror over x axis
        true[:,1] = -true[:,1]
        pred[:,1] = -pred[:,1]
        hist[:,1] = -hist[:,1]

        # find out of bounds points and delete
        true_idxs = []
        for i in range(true.shape[0]):
            if true[i, 0] < xlim[0] or true[i, 0] > xlim[1] or \
               true[i, 1] < ylim[0] or true[i, 1] > ylim[1]:
                true_idxs.append(i)

        pred_idxs = []
        for i in range(pred.shape[0]):
            if pred[i, 0] < xlim[0] or pred[i, 0] > xlim[1] or \
               pred[i, 1] < ylim[0] or pred[i, 1] > ylim[1]:
                pred_idxs.append(i)

        hist_idxs = []
        for i in range(hist.shape[0]):
            if hist[i, 0] < xlim[0] or hist[i, 0] > xlim[1] or \
               hist[i, 1] < ylim[0] or hist[i, 1] > ylim[1]:
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
        ax.plot(hist[:,0], hist[:,1], 'o-', color='r', alpha=0.2, label='history')

        # plot gt
        ax.plot(true[:,0], true[:,1], 'o-', color='g', alpha=0.4, label='true future')

        # plot pred
        ax.plot(pred[:,0], pred[:,1], 'o-', color='b', alpha=0.1, label='prediction')

        # plot connecting points
        ax.plot(true[0,0], true[0,1], 'o', color='k', alpha=1.0, markersize=8., label="current: " + str(ts))

    ax.legend(["history", "true future", "prediction", "current: " + str(ts)], loc="lower right") # FIX LEGEND

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
    for ts in trajectories:
        imgs.append(gen_gif_img(ts))

    gif.save(imgs, outpath, duration=300)

def plotting(trajectories):
    gen_frame(trajectories)
    # TODO(abbielee): fix legend
    plt.show()
