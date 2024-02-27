import glob
import os
import sys

sys.path.append("/usr0/home/tkhot/git_repos/core3d/")
from core3d.mesh import mesh_util
from core3d.texture_mapping import merge
from core3d.texture_mapping.TextureMapper import TextureMapper
import ipdb

# for aoi_name in ['wpafb_d1', 'wpafb_d2','ucsd_d3', 'jacksonville_d4']:
#     print("Constructing DSM from Primitives for AOI : ", aoi_name)
#     ply_list = glob.glob(os.path.join('../../outputs/{}/ply_primitives','{}*.ply'.format(aoi_name, aoi_name)))
#     dtm_file = os.path.join('../../scoring/data/outputs/{}/buildings_dsm'.format(aoi_name), 'dtm.tif')
#     prim_dhm_file = os.path.join('../../scoring/data/outputs/{}/buildings_dsm'.format(aoi_name), 'dhm.tif')
#     prim_dsm_file = os.path.join('../../scoring/data/outputs/{}/buildings_dsm'.format(aoi_name), 'dsm.tif')
#     mesh_util.ply_list_to_dsm(ply_list, dtm_file, prim_dhm_file, prim_dsm_file)


bldg_prim_dir = (
    "/usr0/home/tkhot/Downloads/results/results/jacksonville_d4/buildings_prim"
)
bridge_prim_dir = (
    "/usr0/home/tkhot/Downloads/results/results/jacksonville_d4/bridge_prim"
)

for aoi_name in ["wpafb_d1", "wpafb_d2", "ucsd_d3", "jacksonville_d4"]:

    # add textures
    # bldg_fit_dir = os.path.join(bldg_prim_dir, 'fitting_top_roof')
    bldg_fit_dir = os.path.join("../../outputs/{}/ply_primitives".format(aoi_name))
    # bridge_fit_dir = os.path.join(bridge_prim_dir, 'fitting_top_roof')
    aoi_ply_file = os.path.join(
        "../../outputs/{}/ply_primitives".format(aoi_name), "aoi.ply"
    )
    # scores_dir = os.path.join(bldg_fit_dir, 'scores')
    # os.makedirs(scores_dir, exist_ok=True)

    true_ortho_file = os.path.join(
        "../../scoring/data/outputs/{}/buildings_dsm".format(aoi_name), "true_ortho.tif"
    )
    # aoi_ply_file = os.path.join(scores_dir, 'aoi.ply')
    # buildings_ply_file = os.path.join(scores_dir, 'buildings.ply')
    # bridges_ply_file = os.path.join(scores_dir, 'bridges.ply')
    textured_ply_file = os.path.join(
        "../../outputs/{}/ply_primitives".format(aoi_name), "aoi_textured.ply"
    )

    print("Assembling 3D model")
    merge.merge_plys([bldg_fit_dir], aoi_ply_file)

    print("Texturing 3D model")
    texture_mapper = TextureMapper(aoi_ply_file, true_ortho_file)
    texture_mapper.save(textured_ply_file)

    print("Constructing DSM from Primitives")
    dtm_file = os.path.join(
        "../../scoring/data/outputs/{}/buildings_dsm".format(aoi_name), "dtm.tif"
    )
    prim_dhm_file = os.path.join(
        "../../scoring/data/outputs/{}/buildings_dsm".format(aoi_name), "dhm.tif"
    )
    prim_dsm_file = os.path.join(
        "../../scoring/data/outputs/{}/buildings_dsm".format(aoi_name), "dsm.tif"
    )
    mesh_util.ply_list_to_dsm([aoi_ply_file], dtm_file, prim_dhm_file, prim_dsm_file)
