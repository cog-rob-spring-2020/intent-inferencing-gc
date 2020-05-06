"""
16.412 Intent Inference GC | plotting.py
Plots results of running CVM. Adapted from plotting script received from
Christoph Scholler.
Author: Abbie Lee (abbielee@mit.edu)
"""
import matplotlib.pyplot as plt

class FrameTrajectories:
    def __init__(self, frameID, agentID, gt, pred, hist):
        self.frameID = frameID
        self.true_futures = gt
        self.predicteds = pred
        self.histories = hist

def plotting(trajectories):
    # output_folder = os.path.join("./plots", dataset_name)
    tr = trajectories[12]
    true = tr["true"]
    pred = tr["predicted"]

    print("True Size: ", true.size())
    print("True:",true[:,0], true[:,1])

    print("Pred Size: ", pred.size())
    print("Predicted:", pred[:,0], pred[:,1])

    # plot observed_history
    # scene_obs = obs_traj.permute(1, 0, 2)[seq_start:seq_end]
    # for i, traj in enumerate(scene_obs):
    #     traj = traj.cpu().numpy()
    #     plt.plot(traj[:,0], traj[:,1], 'o-', color='grey', alpha=0.2, label=(None if i > 0 else 'history'))

    true = tr["true"]
    pred = tr["predicted"]
    # plot gt
    plt.plot(true[:,0], true[:,1], 'o-', color='g', alpha=0.4, label='ground truth')

    # plot pred
    plt.plot(pred[:,0], pred[:,1], 'o-', color='red', alpha=0.1, label='prediction')

    # plot connecting points
    # points_t = scene_obs[:,-1,:].cpu().numpy()
    # plt.plot(points_t[:,0], points_t[:,1], 'o', color='darkblue', alpha=0.6, markersize=8., label="timestep t")

    # plt.show()
    plt.rcParams["figure.figsize"] = (7,6)
    plt.tick_params(labelsize=14)
    plt.tick_params(labelsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Y", fontsize=16)
    plt.show()
    # plt.savefig(os.path.join(output_folder, "{}-{}.png".format(batch_id, j)), bbox_inches='tight', pad_inches=0)
    # plt.close()
    #
    # plt.figure()
    # plt.plot(tr["true"][:,0], tr["true"][:,1], 'g')
    # plt.plot(tr["predicted"][:,0], tr["predicted"][:,1], 'r')
    # plt.show()
