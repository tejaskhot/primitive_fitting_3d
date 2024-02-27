import os
import glob


def obj_to_ply(obj_file, ply_file):
    cmd = "meshlabserver -i {} -o {} ".format(obj_file, ply_file)
    os.system(cmd)


for aoi_name in ["wpafb_d1", "wpafb_d2", "ucsd_d3", "jacksonville_d4"]:
    obj_primitives = glob.glob(
        os.path.join(
            "../../outputs/{}/obj_primitives".format(aoi_name),
            "{}*.obj".format(aoi_name),
        )
    )
    obj_overlayed = glob.glob(
        os.path.join(
            "../../outputs/{}/obj_overlayed".format(aoi_name),
            "{}*.obj".format(aoi_name),
        )
    )

    for i, f in enumerate(obj_primitives):
        obj_to_ply(
            f, "../../outputs/{}/ply_primitives/{}_{}.ply".format(aoi_name, aoi_name, i)
        )

    for i, f in enumerate(obj_overlayed):
        obj_to_ply(
            f, "../../outputs/{}/ply_overlayed/{}_{}.ply".format(aoi_name, aoi_name, i)
        )
