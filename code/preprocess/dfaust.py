import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from __future__ import print_function
import argparse

import trimesh
from trimesh.sample import sample_surface
import os
import numpy as np
import json
import utils.general as utils

SAMPLES = 250000


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', required=True, type=str, help='abs path of dfaust data directory')
    parser.add_argument('--out-path', required=True, type=str, help='abs path of parent of output directory')
    parser.add_argument('--mode', required=False, default=None, type=int, help='0 for train only 1 for test only')
    parser.add_argument('--names', required=False, default=None, type=str, help='format: 50002,50020,50021; o.w. all')
    parser.add_argument('--skip', default=False, action="store_true", help='skip if already exists')

    code_path = os.path.abspath(os.curdir)

    print("running")
    args = parser.parse_args()

    if args.mode == 0:

        modes = ['train']

    elif args.mode == 1:

        modes = ['test']

    else:

        modes = ['train', 'test']

    if args.names:
        names = args.names.split(',')
    else:
        names = None

    for mode in modes:

        print('starting preprocessing {0} mode'.format(mode))

        split_file = os.path.join(code_path, 'splits', 'dfaust', '{0}_all.json'.format(mode))
        with open(split_file, "r") as f:
            train_split = json.load(f)

        countera = 0
        counterb = 0
        counterc = 0
        scale = 1

        # os.chdir('/home/atzmonm/data/')
        for ds,cat_det in train_split['scans'].items():
            if names and ds not in names:
                continue
            print("ds :{0} , a:{1}".format(ds,countera))
            countera = countera + 1
            counterb = 0

            for cat,shapes in cat_det.items():
                print("cat {0} : b{1}".format(cat,counterb))
                counterb = counterb + 1
                source = os.path.abspath(os.path.join(args.src_path, 'scans', ds, cat))
                output = os.path.abspath(os.path.join(args.out_path, 'dfaust_processed'))
                utils.mkdir_ifnotexists(output)
                utils.mkdir_ifnotexists(os.path.join(output, ds))
                utils.mkdir_ifnotexists(os.path.join(output, ds, cat))
                counterc = 0

                for item,shape in enumerate(shapes):
                    print("item {0} : c{1}".format(cat, counterc))
                    counterc = counterc + 1
                    output_file = os.path.join(output,ds,cat,shape)
                    print (output_file)
                    if not (args.skip and os.path.isfile(output_file + '.npy')):
                        print ('loading : {0}'.format(os.path.join(source,shape)))
                        mesh = trimesh.load(os.path.join(source,shape) + '.ply')
                        sample = sample_surface(mesh,SAMPLES)
                        pnts = sample[0]
                        normals = mesh.face_normals[sample[1]]
                        center = np.mean(pnts, axis=0)

                        pnts = pnts - np.expand_dims(center, axis=0)
                        point_set = np.hstack([pnts, normals])

                        np.save(output_file + '.npy', point_set)

                        np.save(output_file + '_normalization.npy', {"center":center,"scale":scale})

    print ("end!")
