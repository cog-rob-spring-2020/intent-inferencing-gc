import os
import argparse
import json

import numpy as np
import torch.utils.data as Data

from metrics import *
from ped_dataset import *
from plotting import *


class RunConfig:
    prediction_horizon = 9
    min_sequence_length = 10
    observed_history = 8
    sequence_length = observed_history + prediction_horizon

    sample = False
    num_samples = 20
    sample_angle_std = 25

    use_angvel = False # considers angular velocity in calculation

    make_plot = False
    save_gif = False
    save_imgs = False

    dataset_paths = ["../datasets/CARLA_long",
                     "../datasets/CARLA_short",
                     "../datasets/leftturn",
                     "../datasets/rightturn"]
    dataset_paths = ["../datasets/CARLA_long"]

def rel_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def constant_velocity_model(observed, headings=None, sample=False,
        from_headings=False, average_thetas=False, damping_factor=0.95):
    """
    CVM can be run with or without sampling. A call to this function always
    generates one sample if sample option is true.

    from_headings: Use headings to generate angular velocities (otherwise use
    past few positions)
    average_thetas: Use average of last two theta differences (otherwise
    extrapolate from last theta difference alone)
    damping_factor: Discount theta by a factor of damping_factor**i on the
    ith future timestep
    """
    obs_rel = observed[1:] - observed[:-1]
    deltas = obs_rel[-1].unsqueeze(0)
    if RunConfig.use_angvel:
        if from_headings:
            dthetas = headings[1:] - headings[:-1]
            theta1 = dthetas[-1]
            theta2 = dthetas[-2]
        else:
            v1 = obs_rel[-1].unsqueeze(0)
            v2 = obs_rel[-2].unsqueeze(0)
            v3 = obs_rel[-3].unsqueeze(0)
            theta1 = np.arctan2(v1[:,:,1], v1[:,:,0]) - \
                     np.arctan2(v2[:,:,1], v2[:,:,0])
            theta2 = np.arctan2(v2[:,:,1], v2[:,:,0]) - \
                     np.arctan2(v3[:,:,1], v3[:,:,0])
        if average_thetas:
            theta = (theta1 + theta2)/2
        else:
            theta = theta1

        # normalize to range (-pi, pi]
        if theta > np.pi:
            theta -= 2*np.pi
        elif theta <= -np.pi:
            theta += 2*np.pi
        # rotation matrix is transposed since deltas are a row vector
        # [x y] @ R^T = R @ [x \\ y]
        get_rotation_mat = lambda theta: torch.Tensor(
                [[[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]]])
        y_pred_rel = [deltas @ get_rotation_mat(theta*i*(damping_factor**i))
                for i in range(1, RunConfig.prediction_horizon+1)]
        y_pred_rel = torch.cat(y_pred_rel, 0)
    else:
        y_pred_rel = deltas.repeat(RunConfig.prediction_horizon, 1, 1)
    if sample:
            sampled_angle = np.random.normal(0, RunConfig.sample_angle_std, 1)[0]
            theta = (sampled_angle * np.pi)/ 180.
            c, s = np.cos(theta), np.sin(theta)
            rotation_mat = torch.tensor([[c, s],[-s, c]])
            deltas = torch.t(rotation_mat.matmul(torch.t(deltas.squeeze(dim=0)))).unsqueeze(0)
    return y_pred_rel

def evaluate_testset(testset):
    testset_loader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=False)

    with torch.no_grad():

        avg_displacements =  []
        final_displacements = []
        ts_to_trajectories = {} # timestamp to ID for all the trajectories with that timestamp

        for seq_id, (batch_x, batch_y) in enumerate(testset_loader):
            timestamp = testset.samples[seq_id].trajectory.timestamps[RunConfig.observed_history-1]

            obs_pos, obs_headings = batch_x
            observed = obs_pos.permute(1, 0, 2)
            history = observed.permute(1, 0, 2)
            headings = obs_headings.permute(1, 0)
            y_true_rel, masks = batch_y
            y_true_rel = y_true_rel.permute(1, 0, 2)


            sample_avg_disp = []
            sample_final_disp = []
            samples_to_draw = RunConfig.num_samples if RunConfig.sample else 1
            for i in range(samples_to_draw):

                # predict and convert to absolute
                if RunConfig.use_angvel:
                    y_pred_rel = constant_velocity_model(observed, headings, \
                            sample=RunConfig.sample)
                else:
                    y_pred_rel = constant_velocity_model(observed, \
                            sample=RunConfig.sample)
                y_pred_abs = rel_to_abs(y_pred_rel, observed[-1])
                predicted_positions = y_pred_abs.permute(1, 0, 2)

                # convert true label to absolute
                y_true_abs = rel_to_abs(y_true_rel, observed[-1])
                true_positions = y_true_abs.permute(1, 0, 2)

                # compute errors
                avg_displacement = avg_disp(predicted_positions, [true_positions, masks])
                final_displacement = final_disp(predicted_positions, [true_positions, masks])
                sample_avg_disp.append(avg_displacement)
                sample_final_disp.append(final_displacement)

                if len(RunConfig.dataset_paths) == 1:
                    trajectories = {"observed": history, \
                                    "predicted": predicted_positions, \
                                    "true": true_positions, "ts": timestamp}

                    if timestamp not in ts_to_trajectories.keys():
                        ts_to_trajectories[timestamp] = []

                    ts_to_trajectories[timestamp].append(trajectories)

            avg_displacement = min(sample_avg_disp)
            final_displacement = min(sample_final_disp)
            avg_displacements.append(avg_displacement)
            final_displacements.append(final_displacement)

        print("Total:", seq_id)
        avg_displacements = np.mean(avg_displacements)
        final_displacements = np.mean(final_displacements)

        return avg_displacements, final_displacements, ts_to_trajectories # dID_to_trajectories

def load_datasets():
    datasets = []
    datasets_size = 0
    for dataset_path in RunConfig.dataset_paths:
        dataset_path = dataset_path.replace('~', os.environ['HOME'])
        print("Loading dataset {}".format(dataset_path))
        dataset = PedDataset(dataset_path=dataset_path, sequence_length=RunConfig.sequence_length, observed_history=RunConfig.observed_history, \
                              min_sequence_length=RunConfig.min_sequence_length)
        datasets.append(dataset)
        datasets_size += len(dataset)
    print("Size of all datasets: {}".format(datasets_size))
    return datasets

def parse_commandline():
    parser = argparse.ArgumentParser(description='Runs an evaluation of the Constant Velocity Model.')
    parser.add_argument('--sample', default=RunConfig.sample, action='store_true', help='Turns on the sampling for the CVM (OUR-S).')
    parser.add_argument('--make_plot', default=RunConfig.make_plot, action="store", help='Generate plot for specified timestamp')
    parser.add_argument("--use_angvel", default=RunConfig.use_angvel, action="store_true", help="Use angular velocity in prediction if available.")
    parser.add_argument("--save_gif", default=RunConfig.save_gif, action="store", help="Save gif to fname.")
    parser.add_argument("--save_imgs", default=RunConfig.save_imgs, action="store", help="Save gif frames to dirpath.")
    args = parser.parse_args()
    return args

def main():
    args = parse_commandline()
    RunConfig.sample = args.sample
    RunConfig.make_plot = float(args.make_plot)
    RunConfig.save_gif = args.save_gif
    RunConfig.save_imgs = args.save_imgs
    RunConfig.use_angvel = args.use_angvel
    if RunConfig.sample:
        print("Sampling activated.")
    if RunConfig.make_plot != False:
        print("Plotting activated for timestamp " + str(RunConfig.make_plot) + ".")
    if RunConfig.save_gif != False:
        print("Saving GIF to " + RunConfig.save_gif)
    if RunConfig.save_imgs != False:
        print("Saving images for GIF to " + RunConfig.save_imgs + ". ")
    if RunConfig.use_angvel:
        print("Using angular velocity.")

    print("--------------------------")

    datasets = load_datasets()
    testset_results = []
    for i, testset in enumerate(datasets):
        print("Evaluating testset {}".format(testset.name))
        avg_displacements, final_displacements, ts_to_trajectories = evaluate_testset(testset)
        testset_results.append([testset.name, avg_displacements, final_displacements])

    print("\n== Results for testset evaluations ==")
    total_avg_disp, total_final_disp = 0, 0
    for name, avg_displacements, final_displacements in testset_results:
        print("- Testset: {}".format(name))
        print("ADE: {}".format(avg_displacements))
        print("FDE: {}".format(final_displacements))
        total_avg_disp += avg_displacements
        total_final_disp += final_displacements
    print("- Average")
    print("*ADE: {}".format(total_avg_disp/len(testset_results)))
    print("*FDE: {}".format(total_final_disp/len(testset_results)))

    if len(RunConfig.dataset_paths) == 1:
        if RunConfig.make_plot:
            trajectories = ts_to_trajectories[RunConfig.make_plot]
            plotting(trajectories)

        if RunConfig.save_gif:
            trajectories = [ts_to_trajectories[i] for i in sorted(ts_to_trajectories.keys())]
            plotting_gif(trajectories, RunConfig.save_gif)

        if RunConfig.save_imgs != False:
            trajectories = [ts_to_trajectories[i] for i in sorted(ts_to_trajectories.keys())]

            try:
                os.mkdir(RunConfig.save_imgs)
            except:
                pass
            plotting_saveimgs(trajectories, RunConfig.save_imgs)

if __name__ == "__main__":
    main()
