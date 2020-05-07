"""
16.412 Intent Inference GC | plotting.py
Plots results of running CVM. Adapted from plotting script received from
Christoph Scholler.
Author: Abbie Lee (abbielee@mit.edu)
"""
import matplotlib.pyplot as plt
import imageio

def plotting(trajectories):
    # output_folder = os.path.join("./plots", dataset_name)
    print(len(trajectories))

    for tr in [trajectories[0], trajectories[30]]:
        true = tr["true"][0]
        pred = tr["predicted"][0]
        hist = tr["observed"][0]

        # plot observed_history
        plt.plot(hist[:,0], hist[:,1], 'o-', color='grey', alpha=0.2, label='history')

        # plot gt
        plt.plot(true[:,0], true[:,1], 'o-', color='g', alpha=0.4, label='ground truth')

        # plot pred
        plt.plot(pred[:,0], pred[:,1], 'o-', color='red', alpha=0.1, label='prediction')

        # plot connecting points
        plt.plot(true[0,0], true[0,1], 'o', color='darkblue', alpha=0.6, markersize=8., label="timestep t")

    plt.rcParams["figure.figsize"] = (7,6)
    plt.tick_params(labelsize=14)
    plt.tick_params(labelsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Y", fontsize=16)
    plt.xlim(98, 170)
    plt.ylim(170, 230)
    plt.show()
    # plt.savefig(os.path.join(output_folder, "{}-{}.png".format(batch_id, j)), bbox_inches='tight', pad_inches=0)
    # plt.close()

def gen_plots(trajectories):
    pass
