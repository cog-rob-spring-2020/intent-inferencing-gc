"""
16.412 Intent Inference GC | plotting.py
Plots results of running CVM. Adapted from plotting script received from
Christoph Scholler.
Author: Abbie Lee (abbielee@mit.edu)
"""
import numpy as np
import matplotlib.pyplot as plt
import gif

def gen_frame(detection_trajs):

    xlims = [98, 170]
    ylims = [170, 230]

    fig, ax = plt.subplots()
    # plt.rcParams["figure.figsize"] = (7,6)
    ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])

    for trajectory in detection_trajs:
        true = trajectory["true"][0]
        pred = trajectory["predicted"][0]
        hist = trajectory["observed"][0]
        ts = trajectory["ts"]

        # find out of bounds points and delete
        idxs = []
        for i in range(true.shape[0]):
            if true[i, 0] < xlims[0] or true[i, 0] > xlims[1] or \
               true[i, 1] < ylims[0] or true[i, 1] > ylims[1]:
                idxs.append(i)

        true = np.delete(true, idxs, axis=0)

        # plot observed_history
        ax.plot(hist[:,0], hist[:,1], 'o-', color='grey', alpha=0.2, label='history')

        # plot gt
        ax.plot(true[:,0], true[:,1], 'o-', color='g', alpha=0.4, label='ground truth')

        # plot pred
        ax.plot(pred[:,0], pred[:,1], 'o-', color='red', alpha=0.1, label='prediction')

        # plot connecting points
        ax.plot(true[0,0], true[0,1], 'o', color='darkblue', alpha=0.6, markersize=8., label="timestep " + str(ts))

    ax.legend() # FIX LEGEND

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
