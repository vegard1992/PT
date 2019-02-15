import numpy as np

def flatten_tris(tris):
    vertices = np.zeros((len(tris)*3, 4), dtype=np.float32)
    vertices[:len(tris)*3][::3,:3] = tris[::,0]
    vertices[:len(tris)*3][1::3,:3] = tris[::,1]
    vertices[:len(tris)*3][2::3,:3] = tris[::,2]

    return vertices
