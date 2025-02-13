import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def main():
    project_root = os.environ.get('FORTRAN_ML_ROOT', -1337)
    if project_root == -1337:
        print("Error: Error: Please set the environment variable FORTRAN_ML_ROOT to the root directory of the FORTRAN-ML-INTERFACE project.")

    input_dir = project_root + "/data/"
    out_dir = project_root + "/visualization/results"
    data_files = [f for f in os.listdir(input_dir) if f.endswith("prate_inferred.h5")]
    print(data_files)

    for data_file in data_files:
        print(data_file)
        time = data_file.split("_")[1]

        data_file = input_dir + '/' + data_file
        f = h5py.File(data_file, 'r')

        fig, ax = plt.subplots(1, 4, figsize=(15, 3))
        #plt.suptitle(f"Comparison at time {time}")
        

        flowvars = ['H2O', 'source_H2O', 'omega']
        titles = ["Progress variable (H2O)", "Omega H2O (ground truth)", "Omega H2O (UNet prediction)", "Relative Error"]

        for i, flowvar in enumerate(flowvars):
            dset = f[f"/sd_1d_test/data/cv_data_real/{flowvar}"]
            shape = dset.shape
            print(shape)

            dset_2D = np.reshape(dset, (shape[1], shape[2]))

            
            #ax.set_xlabel("cells in x-direction")
            #ax.set_ylabel("cells in y-direction")
            cols = shape[1]
            rows = shape[2]
            c = ax[i].pcolormesh(range(rows+1), range(cols+1), dset_2D)#, vmin=0, vmax=0.1)
            cb = fig.colorbar(c)
            cb.set_label(f"{flowvar}")
            ax[i].title.set_text(titles[i])

        # add absolute difference field
        omega_true_dset = f[f"/sd_1d_test/data/cv_data_real/source_H2O"]
        omega_pred_dset = f[f"/sd_1d_test/data/cv_data_real/omega"]
        omega_diff = np.abs((omega_true_dset[:] - omega_pred_dset[:])) / np.abs(np.max(omega_true_dset[:]))
        print(f"Min. rel. err = {np.min(omega_diff)}, max. rel. err = {np.max(omega_diff)}")
        shape = np.shape(omega_diff)
        print(shape)
        omega_diff_2D = np.reshape(omega_diff, (shape[1], shape[2]))
        cols = shape[1]
        rows = shape[2]
        cmap = plt.colormaps['bwr']
        c = ax[3].pcolormesh(range(rows+1), range(cols+1), omega_diff_2D, cmap=cmap, vmin=-0.02, vmax=0.02)
        cb = fig.colorbar(c)
        cb.set_label(f"rel. err. [%]")
        ax[3].title.set_text(titles[3])

            
        fig.tight_layout()
        fig.savefig(f"{out_dir}/comparison-{time}.png", format='png')


if __name__ == "__main__":
    main()