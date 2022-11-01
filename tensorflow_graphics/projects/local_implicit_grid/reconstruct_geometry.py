# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Reconstruct scene using LIG.
"""


import os
import sys
import warnings

from absl import app
from absl import flags
import numpy as np

from tensorflow.compat.v1.io import gfile
# from tensorflow_graphics.projects.local_implicit_grid.core import point_utils as pu
# from tensorflow_graphics.projects.local_implicit_grid.core import postprocess
# from tensorflow_graphics.projects.local_implicit_grid.core import reconstruction as rec
from core import point_utils as pu
from core import postprocess
from core import reconstruction as rec
import trimesh

import open3d as o3d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# flags.DEFINE_string('input_ply', '', 'Input point sample ply file.')
# flags.DEFINE_string('output_ply', '', 'Reconstructed scene ply file.')
# flags.DEFINE_integer('steps', 10000, 'Number of optimization steps.')
# flags.DEFINE_integer('npoints', 10000,
#                      'Number of points to sample per iteration during optim.')
# flags.DEFINE_float('part_size', 0.25, 'Size of parts per cell (meters).')
# flags.DEFINE_float('init_std', 0.02, 'Initial std to draw random code from.')
# flags.DEFINE_integer('res_per_part', 0,
#                      'Evaluation resolution per part. A higher value produces a'
#                      'finer output mesh. 0 to use default value. '
#                      'Recommended value: 8, 16 or 32.')
# flags.DEFINE_boolean('overlap', True, 'Use overlapping latent grids.')
# flags.DEFINE_boolean('postprocess', True, 'Post process to remove backfaces.')
# flags.DEFINE_string('ckpt_dir', 'pretrained_ckpt',
#                     'Checkpoint directory.')
#
# FLAGS = flags.FLAGS



def reconstruct(args):
  # if len(argv) > 1:
  #   raise app.UsageError('Too many command-line arguments.')
  if not args.input_ply:
    raise IOError('--input_ply must be specified.')
  if not args.output_ply:
    args.output_ply = args.input_ply.replace('.ply', '.reconstruct.ply')

  # load point cloud from ply file
  # v, n = pu.read_point_ply(args.input_ply)

  if args.input_ply[-3:] == "ply":
    pc = o3d.io.read_point_cloud(args.input_ply)
    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)
    data = {}
    data["points"] = points
    data["normals"] = normals
  else:
    data = np.load(args.input_ply)
  v = data["points"]
  if(args.mode == "mst"):
    n = data["normals"]
    n = n / np.linalg.norm(n, axis=1)[:, np.newaxis]
  elif(args.mode == "jet_sensor"):
    n = data["normals"]
    n = n / np.linalg.norm(n, axis=1)[:, np.newaxis]
    sensor_vec = data["sensor_position"] - v
    ip = np.einsum('ij,ij->i', n, sensor_vec)
    n[ip < 0] = -n[ip < 0]
  elif(args.mode == "svn"):
    if "sensors" in data:
      s = data["sensors"]
    else:
      s = data["sensor_position"]
    n = s - v
    n = n / np.linalg.norm(n, axis=1)[:, np.newaxis]
  elif(args.mode == "gtn"):
    n = data["gt_normals"]
    n = n / np.linalg.norm(n, axis=1)[:, np.newaxis]
  else:
    print("{} is not a valid normal mode".format(args.mode))
    sys.exit(1)

  # idx=np.random.randint(v.shape[0], size=3000)
  # v=v[idx]
  # n=n[idx]


  # check if part size is too large
  min_bb = np.min(np.max(v, axis=0) - np.min(v, axis=0))
  if args.part_size > 0.25 * min_bb:
    warnings.warn(
        'WARNING: part_size seems too large. Recommend using a part_size < '
        '{:.2f} for this shape.'.format(0.25 * min_bb), UserWarning)

  surface_points = np.concatenate([v, n], axis=1)
  near_surface_samples = rec.get_in_out_from_ray(
      surface_points, sample_factor=10, std=0.01)

  xmin = np.min(surface_points[:, :3], 0)
  xmax = np.max(surface_points[:, :3], 0)

  # add some extra slack to xmin and xmax
  xmin -= args.part_size
  xmax += args.part_size

  if args.res_per_part == 0:
    res_per_part = int(64*args.part_size)
  else:
    res_per_part = args.res_per_part
  npts = min(near_surface_samples.shape[0], args.npoints)-1

  print('Performing latent grid optimization...')
  v, f, _, _ = rec.encode_decoder_one_scene(
      near_surface_samples, args.ckpt_dir, args.part_size, overlap=True,
      indep_pt_loss=True, init_std=args.init_std,
      xmin=xmin, xmax=xmax, res_per_part=res_per_part,
      npts=npts, steps=args.steps)

  out_dir = os.path.dirname(args.output_ply)
  if out_dir and not gfile.exists(out_dir):
    gfile.makedirs(out_dir)
  mesh = trimesh.Trimesh(v, f)

  if args.postprocess:
    print('Postprocessing generated mesh...')
    mesh = postprocess.remove_backface(mesh, surface_points)

  print('Writing reconstructed mesh to {}'.format(args.output_ply))
  with gfile.GFile(args.output_ply, 'wb') as fh:
    mesh.export(fh, 'ply')

if __name__ == '__main__':
  app.run(reconstruct)
