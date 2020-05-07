"""
16.412 Intent Inference GC | plotting.py
Plots results of running CVM. Adapted from plotting script received from
Christoph Scholler.
Author: Abbie Lee (abbielee@mit.edu)
"""
import numpy as np
import matplotlib.pyplot as plt
import gif

@gif.frame
def gen_img(trajectory):
    # output_folder = os.path.join("./plots", dataset_name)

    true = trajectory["true"][0]
    pred = trajectory["predicted"][0]
    hist = trajectory["observed"][0]

    # find (0,0) and delete
    idxs = []
    for i in range(true.shape[0]):
        if true[i,:] == [0, 0]:
            idxs.append(i)

    true = np.delete(true, idxs, axis=0)

    fig, ax = plt.subplots()
    # plt.rcParams["figure.figsize"] = (7,6)
    ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_xlim(98, 170)
    ax.set_ylim(170, 230)

    # plot observed_history
    ax.plot(hist[:,0], hist[:,1], 'o-', color='grey', alpha=0.2, label='history')

    # plot gt
    ax.plot(true[:,0], true[:,1], 'o-', color='g', alpha=0.4, label='ground truth')

    # plot pred
    ax.plot(pred[:,0], pred[:,1], 'o-', color='red', alpha=0.1, label='prediction')

    # plot connecting points
    ax.plot(true[0,0], true[0,1], 'o', color='darkblue', alpha=0.6, markersize=8., label="timestep t")

    ax.legend()

    # fig.canvas.draw()
    # image = np.frombuffer(fig.canvas.tostring_rgb()) #dtype='uint8')
    # # image = Image.fromarray(im_array, "RGB")
    # image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # ims.append(image)

def plotting(trajectories):
    imgs = []
    for tr in trajectories:
        imgs.append(gen_img(tr))

    gif.save(imgs, "./cvm.gif", duration=150)

    # kwargs_write = {'fps':3.0, 'quantizer':'nq'}
    # imageio.mimsave('./cvm.gif', images, fps=1)
    # plt.savefig(os.path.join(output_folder, "{}-{}.png".format(batch_id, j)), bbox_inches='tight', pad_inches=0)
    # plt.close()
