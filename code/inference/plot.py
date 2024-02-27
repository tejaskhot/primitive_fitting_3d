import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from obj_utils import *
from plyfile import PlyData

cmap = cm.get_cmap("tab10")
colors = [cmap(i) for i in np.arange(0, 1, 1 / 10)]


def euler2rot(theta):
    """Return a 3D rotation matrix given a vector of angles along (X,Y,Z)"""
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def Rotate2D(pts, cnt, ang=np.pi / 4):
    """pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian"""
    return (
        np.dot(
            pts - cnt,
            np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]),
        )
        + cnt
    )


def get_processed_cubes(cubes, return_idx=False):
    """remove cubes which lie entirely inside of another cube"""
    idx = []
    out = []
    for c1, P1 in enumerate(cubes):
        flag = True
        for c2, P2 in enumerate(cubes):
            if c1 == c2:
                continue
            i = P1[0] - P1[4]
            j = P1[5] - P1[4]
            k = P1[7] - P1[4]
            v = P2 - P1[4]
            counts = (
                (0 <= np.dot(v, i))
                & (np.dot(v, i) <= np.dot(i, i))
                & (0 <= np.dot(v, j))
                & (np.dot(v, j) <= np.dot(j, j))
                & (0 <= np.dot(v, k))
                & (np.dot(v, k) <= np.dot(k, k))
            )
            if sum(counts) == 8:
                flag = False
                idx.append(c2)
    out = [cubes[i] for i in range(len(cubes)) if i not in idx]
    if len(out) < len(cubes):
        print(
            "Removed {} cubes for being within another cube".format(
                len(cubes) - len(out)
            )
        )
    if return_idx:
        return out, idx
    return out


def draw_cube_points(
    cubes=None, pts=None, save=False, save_path="", save_gif=False, aoi_name=""
):

    def animate(i):
        ax.view_init(elev=30, azim=i)
        return (fig,)

    fig = plt.figure()
    ax = Axes3D(fig)
    minmax = {}
    minmax["xmin"] = 1e15
    minmax["ymin"] = 1e15
    minmax["zmin"] = 1e15
    minmax["xmax"] = -1e15
    minmax["ymax"] = -1e15
    minmax["zmax"] = -1e15

    cube_cords = []
    if cubes is not None:
        if len(cubes.shape) == 1:
            cubes = cubes.reshape((1, -1))
        for i, cube in enumerate(cubes):
            if len(cube) > 6:
                R = euler2rot([0, 0, cube[-1]])
            else:
                R = euler2rot([0, 0, 0])
            R = R.T
            S = np.diag(cube[3:6] * 2)

            # unit cube at center
            P1 = np.array(
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ]
            )
            # apply transforms
            P2 = np.dot(P1, S)
            P = np.dot(P2, R) + cube[0:3]

            # add offset to heights based on points
            zmin = pts.min(0)[-1]
            diff = P.min(0)[-1] - zmin
            P[:, -1][:4] -= diff
            cube_cords.append(P)

    for i, P in enumerate(get_processed_cubes(cube_cords)):
        sides = [
            [P[0], P[1], P[2], P[3]],
            [P[4], P[5], P[6], P[7]],
            [P[0], P[1], P[5], P[4]],
            [P[1], P[2], P[6], P[5]],
            [P[4], P[7], P[3], P[0]],
            [P[2], P[3], P[7], P[6]],
        ]
        collection = Poly3DCollection(sides, linewidths=1, alpha=1.0)
        collection.set_edgecolor("k")
        collection.set_facecolor(colors[i])
        ax.add_collection3d(collection)
        minmax["xmin"] = min(minmax["xmin"], P.min(0)[0])
        minmax["ymin"] = min(minmax["ymin"], P.min(0)[1])
        minmax["zmin"] = min(minmax["zmin"], P.min(0)[2])
        minmax["xmax"] = max(minmax["xmax"], P.max(0)[0])
        minmax["ymax"] = max(minmax["ymax"], P.max(0)[1])
        minmax["zmax"] = max(minmax["zmax"], P.max(0)[2])

    if pts is not None:
        ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="r", depthshade=False)

    # Hide grid lines
    ax.grid(False)
    plt.axis("off")
    # Hide axes ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_aspect("equal")
    if save:
        if save_gif:
            # Animate
            anim = animation.FuncAnimation(
                fig, animate, frames=360, interval=20, blit=True
            )
            # Save
            # anim.save('{}.mp4'.format(name), fps=30, extra_args=['-vcodec', 'libx264'])
            name_gif = len(
                [_ for _ in os.listdir(os.path.join(save_path, "gifs")) if "gif" in _]
            )
            anim.save(
                os.path.join(save_path, "gifs/{}_{}.gif".format(aoi_name, name_gif)),
                writer="imagemagick",
                fps=30,
            )
            print(
                "gif saved at : ",
                os.path.join(save_path, "gifs/{}_{}.gif".format(aoi_name, name_gif)),
            )
        name_img = len(
            [_ for _ in os.listdir(os.path.join(save_path, "images")) if "jpg" in _]
        )
        plt.savefig(
            os.path.join(save_path, "images/{}_{}.jpg".format(aoi_name, name_img))
        )
        print(
            "img saved at : ",
            os.path.join(save_path, "images/{}_{}.jpg".format(aoi_name, name_img)),
        )
        plt.close()
    else:
        plt.show()


def plot_obj(points, params, save_path, name_obj="", aoi_name=""):
    """save ply files output"""
    cube_cords = []
    for i, cube in enumerate(params):
        if len(cube) > 6:
            R = euler2rot([0, 0, cube[-1]])
        else:
            R = euler2rot([0, 0, 0])
        R = R.T
        S = np.diag(cube[3:6] * 2)

        # unit cube at center
        P1 = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        # apply transforms
        P2 = np.dot(P1, S)
        P = np.dot(P2, R) + cube[0:3]

        # add offset to heights based on points
        # zmin = points.min(0)[-1]
        # diff = P.min(0)[-1] - zmin
        # P[:,-1][:4] -= diff
        cube_cords.append(P)

    _, idx = get_processed_cubes(cube_cords, return_idx=True)
    params = [params[i] for i in range(len(cube_cords)) if i not in idx]

    vs = []
    fs = []
    mtls = []
    for i, cube in enumerate(params):
        vs1, fs1 = params_to_cubes(cube)
        mtls1 = i + np.zeros(len(fs1)).astype(np.int32)
        if len(fs) > 0:
            fs1 += np.max(fs)
            fs = np.vstack((fs, fs1))
            vs = np.vstack((vs, vs1))
            mtls = np.hstack((mtls, mtls1))
        else:
            vs = np.array(vs1)
            fs = np.array(fs1)
            mtls = mtls1
    cols = [colors[i][:3] for i in range(len(params))]

    # plot only primitives
    if not name_obj:
        name_obj = len(
            [
                _
                for _ in os.listdir(os.path.join(save_path, "obj_primitives"))
                if "obj" in _
            ]
        )
    fout1 = open(
        os.path.join(
            save_path, "obj_primitives", "{}_{}.obj".format(aoi_name, name_obj)
        ),
        "w",
    )
    fout2 = open(
        os.path.join(
            save_path, "obj_primitives", "{}_{}.mtl".format(aoi_name, name_obj)
        ),
        "w",
    )

    append_mtl_obj(fout1, vs, fs, mtls, "{}_{}.mtl".format(aoi_name, name_obj))
    append_mtl(fout2, list(set(mtls)), cols)
    fout1.close()
    fout2.close()
    print(
        "obj primitives saved at : ",
        os.path.join(
            save_path, "obj_primitives", "{}_{}.obj".format(aoi_name, name_obj)
        ),
    )

    vs1, fs1 = points_to_cubes(points, 0.005)
    fs1 += np.max(fs)
    fs = np.vstack((fs, fs1))
    vs = np.vstack((vs, vs1))
    mtls1 = len(params) + np.zeros(len(fs1)).astype(np.int32)
    mtls = np.hstack((mtls, mtls1))
    # cols.append(colors[len(cols)][:3])
    cols.append([0, 0, 0])

    # plot points and primitives overlayed
    if not name_obj:
        name_obj = len(
            [
                _
                for _ in os.listdir(os.path.join(save_path, "obj_overlayed"))
                if "obj" in _
            ]
        )
    fout1 = open(
        os.path.join(
            save_path, "obj_overlayed", "{}_{}.obj".format(aoi_name, name_obj)
        ),
        "w",
    )
    fout2 = open(
        os.path.join(
            save_path, "obj_overlayed", "{}_{}.mtl".format(aoi_name, name_obj)
        ),
        "w",
    )

    append_mtl_obj(fout1, vs, fs, mtls, "{}_{}.mtl".format(aoi_name, name_obj))
    append_mtl(fout2, list(set(mtls)), cols)
    fout1.close()
    fout2.close()
    print(
        "obj (points+primitives) saved at : ",
        os.path.join(
            save_path, "obj_overlayed", "{}_{}.obj".format(aoi_name, name_obj)
        ),
    )


if __name__ == "__main__":
    data = PlyData.read("../../data/synthetic/34184_2.ply")
    data = np.asarray(data["vertex"].data.tolist(), dtype=np.float32)
    cubes = np.load("../../data/params_synthetic.npy").item()
    cube = cubes["34184_2"]

    # center = (data.max(0)+data.min(0))/2
    # data = data - center
    # scale = np.max(data.max(0)-data.min(0))/2
    # data = data/scale

    # cube[:,:3] -= center
    # cube[:,:6] /= scale
    # data[:,:2] = Rotate2D(data[:,:2], [0,0], -cube[0][6])
    # cube[:,:2] = Rotate2D(cube[:,:2], [0,0], -cube[0][6])

    # data = data + center
    # data = data*scale

    # cube[:,:3] += center
    # cube[:,:6] *= scale

    # draw_cube_points(cube, data, save=False, save_path='../../outputs/')
    # plot_obj(data, cube, save_path='../../outputs/')
    # import glob
    # import ipdb
    # for f in glob.iglob('../../data/synthetic/*5.ply'):
    #     data = PlyData.read(f)
    #     data = np.asarray(data['vertex'].data.tolist(), dtype=np.float32)
    #     cubes = np.load('../../data/params_synthetic.npy').item()
    #     cube = cubes[f[21:-4]]
    #     draw_cube_points(cube, data, save=False, save_path='../../outputs/')

    data = PlyData.read("/usr0/home/tkhot/Downloads/slides.ply")
    data = np.asarray(data["vertex"].data.tolist(), dtype=np.float32)
    cube = np.load("/usr0/home/tkhot/Downloads/slides.npy")
    # cube = cubes['34184_2']
    draw_cube_points(cube, data, save=False, save_path="../../outputs/")
