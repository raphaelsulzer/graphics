import argparse
import os, sys
import time
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),"..","..","..","graphics","tensorflow_graphics","projects","local_implicit_grid"))
from reconstruct_geometry import reconstruct
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__),"..","..","..","..","benchmark","datasets"))
from modelnet10 import ModelNet10
from shapenet import ShapeNet
from berger import Berger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('--gpu', type=str, default="0",
                        help='Which gpu to use')

    parser.add_argument('--input_ply',type=str, default='',help='Input point sample ply file.')
    parser.add_argument('--output_ply',type=str,default='', help='Reconstructed scene ply file.')
    parser.add_argument('--steps',type=int, default=3001, help='Number of optimization steps.')
    parser.add_argument('--npoints',type=int, default=2048,
                         help='Number of points to sample per iteration during optim.')
    parser.add_argument('--part_size',type=float, default=0.20, help='Size of parts per cell (meters).')
    parser.add_argument('--init_std', type=float,default=0.02, help='Initial std to draw random code from.')
    parser.add_argument('--res_per_part', type=int,default=0,
                         help='Evaluation resolution per part. A higher value produces a'
                         'finer output mesh. 0 to use default value. '
                         'Recommended value: 8, 16 or 32.')
    parser.add_argument('--overlap', type=bool,default=True, help='Use overlapping latent grids.')
    parser.add_argument('--postprocess', type=bool,default=True, help='Post process to remove backfaces.')
    parser.add_argument('--ckpt_dir',type=str, default='pretrained_ckpt',
                        help='Checkpoint directory.')
    parser.add_argument('--mode', type=str, default="mst",
                        help='The scan conf')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # dataset = ShapeNet()
    # split = "test100"
    # models = dataset.getModels(reduce=0.1,scan='6',splits=[split])[split]
    # outpath = "/mnt/raphael/ShapeNet_out/benchmark/lig/shapenet10000"

    # dataset = Berger()
    # models = dataset.getModels(scan=["4"])
    # outpath = "/mnt/raphael/ShapeNet_out/benchmark/lig/reconbench"

    dataset = Berger()
    models = dataset.getModels(scan_conf=["mvs4"])
    outpath = "/mnt/raphael/reconbench_out/mvs/lig"

    # dataset = ModelNet10()
    # split = "test"
    # models = dataset.getModels(splits=[split], classes=["bathtub", "bed", "desk", "dresser", "nightstand", "toilet"])[split]
    # outpath = "/mnt/raphael/ShapeNet_out/benchmark/lig/modelnet"



    print("Reconstruct shapes to ", outpath)

    for m in tqdm(models,ncols=50):

        args.input_ply = m["scan_ply"]

        # outpath = os.path.join(outpath,str(r),str(s),str(np))
        os.makedirs(os.path.join(outpath,m["class"]),exist_ok=True)
        args.output_ply = os.path.join(outpath, m["class"], m["model"] + ".ply")
        try:
            reconstruct(args)
        except Exception as e:
            raise
            print(e)
            print("Problem with shape {}/{}".format(m["class"],m["model"]))


