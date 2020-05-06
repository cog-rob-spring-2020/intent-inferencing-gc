def plot_scene(dataset_name, batch_id, batch, predicted_trajs):
    predicted_trajs = predicted_trajs.permute(0, 2, 1, 3)

    output_folder = os.path.join("./plots", dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    non_linear_ped, loss_mask, seq_start_end) = batch

    for j, (seq_start, seq_end) in enumerate(seq_start_end):
        # plot history
        scene_obs = obs_traj.permute(1, 0, 2)[seq_start:seq_end]
        for i, traj in enumerate(scene_obs):
            traj = traj.cpu().numpy()
            plt.plot(traj[:,0], traj[:,1], 'o-', color='grey', alpha=0.2, label=(None if i > 0 else 'history'))

        # plot gt
        scene_gt = pred_traj_gt.permute(1, 0, 2)[seq_start:seq_end]
        scene_gt = torch.cat([scene_obs[:,-1:,:], scene_gt], dim=1)
        for i, traj in enumerate(scene_gt):
            traj = traj.cpu().numpy()
            plt.plot(traj[:,0], traj[:,1], 'o-', color='g', alpha=0.4, label=(None if i > 0 else 'ground truth'))

        # plot pred
        scene_pred = predicted_trajs[:,seq_start:seq_end]
        scene_pred = torch.cat([scene_obs[:,-1:,:].repeat(scene_pred.size(0), 1, 1, 1), scene_pred], dim=2)
        scene_pred = scene_pred.contiguous().view(-1, scene_pred.size(2), scene_pred.size(3))
        for i, traj in enumerate(scene_pred):
            traj = traj.cpu().numpy()
            plt.plot(traj[:,0], traj[:,1], 'o-', color='red', alpha=0.1, label=(None if i > 0 else 'prediction'))

        # plot connecting points
        points_t = scene_obs[:,-1,:].cpu().numpy()
        plt.plot(points_t[:,0], points_t[:,1], 'o', color='darkblue', alpha=0.6, markersize=8., label="timestep t")

        # plt.show()
        plt.rcParams["figure.figsize"] = (7,6)
        plt.tick_params(labelsize=14)
        plt.tick_params(labelsize=14)
        plt.tick_params(labelsize=14)
        plt.xlabel("X", fontsize=16)
        plt.ylabel("Y", fontsize=16)
        plt.savefig(os.path.join(output_folder, "{}-{}.png".format(batch_id, j)), bbox_inches='tight', pad_inches=0)
        plt.close()
