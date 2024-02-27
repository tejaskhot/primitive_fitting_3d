import argparse
import os
import sys
import random
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append("../core/")

from model import *
from plot import *
from plyfile import PlyData
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument("--num_points", type=np.int_, default=4096, help="number of points")
parser.add_argument(
    "--num_primitives", type=np.int_, default=5, help="number of cuboids"
)
parser.add_argument(
    "--num_params",
    type=np.int_,
    default=6,
    help="number of parameters to predict per primitive",
)

parser.add_argument(
    "--real",
    default=True,
    type=lambda x: (str(x).lower() == "true"),
    help="use real data",
)
parser.add_argument(
    "--save",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="if false, matplotlib plots are displayed",
)
parser.add_argument(
    "--var_count",
    default=True,
    type=lambda x: (str(x).lower() == "true"),
    help="use variable count prediction",
)
parser.add_argument(
    "--cuda", default=True, type=lambda x: (str(x).lower() == "true"), help="use cuda"
)
parser.add_argument(
    "--save_gif",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="save a gif",
)
parser.add_argument(
    "--save_plot",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="save matplotlib plots",
)

parser.add_argument("--aoi", type=str, default="", help="aoi name")
parser.add_argument(
    "--syn_path", type=str, default="../../data/synthetic/", help="data path"
)
parser.add_argument(
    "--params_path",
    type=str,
    default="../../data/params_synthetic.npy",
    help="params path",
)
parser.add_argument(
    "--file_path", type=str, default="../../data/real/", help="data path"
)
parser.add_argument("--save_path", type=str, default="../../outputs/", help="save path")

parser.add_argument(
    "--model",
    type=str,
    default="../../saved_models/norot_rnn5.pth",
    help="trained model path",
)

opt = parser.parse_args()
print(opt)

model = RLNet(
    num_points=opt.num_points,
    out_size=opt.num_params,
    num_primitives=opt.num_primitives,
)
if opt.cuda:
    model.cuda()

if opt.model != "":
    model.load_state_dict(torch.load(opt.model))
    model.eval()

if opt.real:
    data_folder = opt.file_path
else:
    data_folder = opt.syn_path

if len(opt.aoi) > 0:
    aoi = [opt.aoi]
else:
    # aoi = ['wpafb_d1', 'wpafb_d2','ucsd_d3', 'jacksonville_d4']
    aoi = ["COMMERCIALhotel_building_mesh0461"]

for aoi_name in aoi:
    if not os.path.exists(os.path.join(opt.save_path, aoi_name)):
        os.makedirs(os.path.join(opt.save_path, aoi_name))
        os.makedirs(os.path.join(opt.save_path, aoi_name, "gifs"))
        os.makedirs(os.path.join(opt.save_path, aoi_name, "images"))
        os.makedirs(os.path.join(opt.save_path, aoi_name, "obj_overlayed"))
        os.makedirs(os.path.join(opt.save_path, aoi_name, "obj_primitives"))
        os.makedirs(os.path.join(opt.save_path, aoi_name, "params"))
        os.makedirs(os.path.join(opt.save_path, aoi_name, "ply_primitives"))
        os.makedirs(os.path.join(opt.save_path, aoi_name, "ply_overlayed"))
    print("Processing : ", aoi_name)
    print(glob.glob(os.path.join(data_folder, "{}*.ply".format(aoi_name))))
    for fname in glob.glob(os.path.join(data_folder, "{}*.ply".format(aoi_name))):
        points = PlyData.read(fname)
        print("Processing : ", fname)
        points = np.asarray(points["vertex"].data.tolist(), dtype=np.float32)
        indices = np.random.randint(points.shape[0], size=4096)
        points = points[indices, :3]
        # normalize the point cloud to fit in a unit cube with longest side being unit length
        center = (points.max(0) + points.min(0)) / 2
        points = points - center
        scale = np.max(points.max(0) - points.min(0)) / 2
        points = points / scale

        points = Variable(torch.Tensor(points))
        points = points.unsqueeze(0)
        points = points.transpose(2, 1)
        if opt.cuda:
            points = points.cuda()

        pred, probs = model(points)
        # determine count based on stopping probability
        probs_int = (probs > 0.5).long()
        idx = torch.argmax(probs_int, 1)
        print("=" * 50)
        print("num primitives predicted : ", (idx + 1).data.cpu().numpy()[0])

        pred = torch.reshape(pred, (pred.shape[0], opt.num_primitives, opt.num_params))
        pred = pred.squeeze(0).detach().cpu().numpy()
        points = points.squeeze(0).transpose(0, 1).detach().cpu().numpy()
        if opt.var_count:
            pred = pred[: (idx[0] + 1)]

        # scale result to be in same world coordinates as original
        points = points * scale
        points = points + center
        pred[:, :6] *= scale
        pred[:, :3] += center

        # import pdb
        # pdb.set_trace()
        # make primitives touch the ground

        zmin = np.min(points, 0)
        for i in range(len(pred)):
            diff = pred[i, 2] - zmin[-1]
            pred[i, 2] -= diff

        # save params
        name_par = len(
            [
                _
                for _ in os.listdir(os.path.join(opt.save_path, aoi_name, "params"))
                if "npy" in _
            ]
        )
        np.save(
            os.path.join(
                opt.save_path,
                aoi_name,
                "params",
                "{}_{}.npy".format(aoi_name, name_par),
            ),
            np.array(pred),
        )
        print(
            "params saved at : ",
            os.path.join(
                opt.save_path, "params", "{}_{}.npy".format(aoi_name, name_par)
            ),
        )
        # save obj files
        plot_obj(
            points,
            pred,
            save_path=os.path.join(opt.save_path, aoi_name),
            aoi_name=aoi_name,
        )
        if opt.save_plot:
            # save matplotlib plots
            draw_cube_points(
                pred,
                points,
                save=opt.save,
                save_path=os.path.join(opt.save_path, aoi_name),
                save_gif=opt.save_gif,
                aoi_name=aoi_name,
            )
