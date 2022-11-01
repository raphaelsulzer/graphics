import argparse
import os, sys
import time

import numpy as np
from tqdm import tqdm
from reconstruct_geometry import reconstruct
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    dataset = "ModelNet"
    parser.add_argument('-d', '--dataset_dir', type=str, default="/mnt/raphael/ModelNet10",
                        help='working directory which should include the different scene folders.')
    # parser.add_argument('--overwrite', type=int, default=0,
    #                     help='overwrite existing files')
    # parser.add_argument('--sure_dir', type=str, default="/home/raphael/cpp/surfaceReconstruction/build/release",
    #                     help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('--conf', type=int, default=43,
                        help='The scan conf')

    parser.add_argument('--gpu', type=str, default="0",
                        help='Which gpu to use')
    #
    #
    parser.add_argument('--category', type=str, default=None,
                        help='Indicate the category class')

    parser.add_argument('--input_ply',type=str, default='',help='Input point sample ply file.')
    parser.add_argument('--output_ply',type=str,default='', help='Reconstructed scene ply file.')
    parser.add_argument('--steps',type=int, default=3001, help='Number of optimization steps.')
    parser.add_argument('--npoints',type=int, default=2048,
                         help='Number of points to sample per iteration during optim.')
    parser.add_argument('--part_size',type=float, default=0.10, help='Size of parts per cell (meters).')
    parser.add_argument('--init_std', type=float,default=0.02, help='Initial std to draw random code from.')
    parser.add_argument('--res_per_part', type=int,default=0,
                         help='Evaluation resolution per part. A higher value produces a'
                         'finer output mesh. 0 to use default value. '
                         'Recommended value: 8, 16 or 32.')
    parser.add_argument('--overlap', type=bool,default=True, help='Use overlapping latent grids.')
    parser.add_argument('--postprocess', type=bool,default=True, help='Post process to remove backfaces.')
    parser.add_argument('--ckpt_dir',type=str, default='pretrained_ckpt',
                        help='Checkpoint directory.')
    parser.add_argument('--mode', type=str, default="svn",
                        help='The scan conf')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(args.dataset_dir)


    # scan all training data with random configuration from 0,1,2
    # and test data with 0,1,2

    ### scanner confs
    # 0 (easy) --cameras 15 --points 12000 --noise 0.000 --outliers 0.0
    # 1 (medium) --cameras 15 --points 3000 --noise 0.0025 --outliers 0.0
    # 2 (hard) --cameras 15 --points 12000 --noise 0.005 --outliers 0.33
    # 3 (convonet) --cameras 50 --points 3000 --noise 0.005 --outliers 0.0

    # categories = ["rooms_08"]
    times = []

    if dataset == "ModelNet":
        outdir = "/mnt/raphael/ModelNet10_out"
    elif dataset == "synthetic_room":
        outdir = "/mnt/raphael/synthetic_room_out/5"

    for i,c in enumerate(categories):
        # if c.startswith('.'):
        #     continue
        print("\n\n############## Processing {} - {}/{} ############\n\n".format(c,i+1,len(categories)))

        with open(os.path.join(args.dataset_dir,c,"test.lst"), 'r') as f:
            models = f.read().split('\n')

        for m in tqdm(models,ncols=50):
            if(dataset == "ModelNet"):
                args.input_ply = os.path.join(args.dataset_dir,c,"scan",str(args.conf),m,"scan.npz")
                os.makedirs(os.path.join(outdir, "lig", args.mode, "meshes", c), exist_ok=True)
                args.output_ply = os.path.join(outdir, "lig", args.mode+"_"+str(args.part_size), "meshes", c, m + ".ply")
            elif(dataset == "synthetic_rooms"):
                args.input_ply = os.path.join(args.dataset_dir,c,m,"scan",str(args.conf)+".npz")
                os.makedirs(os.path.join(outdir, str(args.conf), "lig", args.mode, "meshes", c), exist_ok=True)
                args.output_ply = os.path.join("/mnt/raphael/synthetic_room_out", "lig", args.mode, "meshes", c, m + ".ply")
            else:
                print("{} is not a valid dataset".format(dataset))
                sys.exit(1)

            try:
                t0 = time.time()
                reconstruct(args)
                d = time.time() - t0
                timed = {}
                timed["class"] = c
                timed["model"] = m
                timed["time"] = d
                times.append(timed)
            except:
                raise
                print("Problem with shape {}/{}".format(c,m))


    times_df = pd.DataFrame.from_dict(times)
    times_df_class = times_df.groupby(by=['class']).mean()
    times_df_class.loc['mean'] = times_df_class.mean()
    times_df_class.to_csv(os.path.join(outdir, "lig", args.mode+"_"+str(args.part_size), 'times_class.csv'))

    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):  # more options can be specified also
        # print(results_df)
        print(times_df_class)
