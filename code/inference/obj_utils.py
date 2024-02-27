import numpy as np

cube_v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
cube_v = cube_v - 0.5

cube_f = np.array([[1,  7,  5 ], [1,  3,  7 ], [1,  4,  3 ], [1,  2,  4 ], [3,  8,  7 ], [3,  4,  8 ], [5,  7,  8 ], [5,  8,  6 ], [1,  5,  6 ], [1,  6,  2 ], [2,  6,  8 ], [2,  8,  4]]).astype(np.int_)

def append_obj(mf_handle, vertices, faces):
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    for fx in range(faces.shape[0]):
        mf_handle.write('f {:d} {:d} {:d}\n'.format(faces[fx, 0], faces[fx, 1], faces[fx, 2]))
    return

def append_mtl_obj(mf_handle, vertices, faces, mtl_ids, mtl_name):
    mf_handle.write('mtllib {}\n'.format(mtl_name))
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    mts = np.unique(mtl_ids)
    for m in mts:
        faces_m = faces[mtl_ids==m]
        mf_handle.write('usemtl m{}\n'.format(m))
        for fx in range(faces_m.shape[0]):
            mf_handle.write('f {:d} {:d} {:d}\n'.format(faces_m[fx, 0], faces_m[fx, 1], faces_m[fx, 2]))
    return

def append_mtl(mtl_handle, mtl_ids, colors):
    for mx in range(len(mtl_ids)):
        mtl_handle.write('newmtl m{}\n'.format(mtl_ids[mx]))
        # The Kd statement specifies the diffuse reflectivity using RGB values.
        mtl_handle.write('Kd {:f} {:f} {:f}\n'.format(colors[mx][0], colors[mx][1], colors[mx][2]))
        # The Ka statement specifies the ambient reflectivity using RGB values.
        mtl_handle.write('Ka 0 0 0\n')
    return

def points_to_cubes(points, edge_size=0.05):
    '''
    Converts an input point cloud to a set of cubes.

    Args:
        points: N X 3 array
        edge_size: cube edge size
    Returns:
        vs: vertices
        fs: faces
    '''
    v_counter = 0
    tot_points = points.shape[0]
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = np.tile(np.linspace(0, 12*tot_points-1, 12*tot_points), 3).reshape(3, 12*tot_points).transpose()
    f_offset = (f_offset//12 * 8).astype(np.int_)
    f_all += f_offset
    for px in range(points.shape[0]):
        v_all[v_counter:v_counter+8,:] *= edge_size
        v_all[v_counter:v_counter+8,:] += points[px, :]
        v_counter += 8

    return v_all, f_all

def params_to_cubes(params):
    '''
    Generates a cube from a set of parameters

    Args:
        points: N X 3 array
        edge_size: cube edge size
    Returns:
        vs: vertices
        fs: faces
    '''
    tot_points = 1
    v_all = cube_v
    f_all = cube_f
    f_offset = np.tile(np.linspace(0, 12*tot_points-1, 12*tot_points), 3).reshape(3, 12*tot_points).transpose()
    f_offset = (f_offset//12 * 8).astype(np.int_)

    S = np.diag(params[3:6]*2)
    v_all = np.dot(v_all, S)
    v_all += params[:3]
    # print('======\n', f_all)
    return v_all, f_all+f_offset
