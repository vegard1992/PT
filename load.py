try:
    from plyfile import PlyData, PlyElement
    has_plyfile = True
except ImportError:
    has_plyfile = False

import numpy as np
from io import StringIO, BytesIO

def loadply(ply_path):
    return loadply_own(ply_path)

def loadply_plyfile(ply_path):
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(f)

    vertices = None
    faces = None
    for e in plydata.elements:
        if e.name == "vertex":
            vertices = e
        if e.name == "face":
            faces = e

    triangles = []

    for f in faces.data:
        i1, i2, i3 = f[0]
        p1 = vertices.data[i1]
        p2 = vertices.data[i2]
        p3 = vertices.data[i3]

        triangles.append((p1, p2, p3))

    return triangles

def loadply_own(ply_path):
    # first determine if its binary!
    with open(ply_path, "rb") as f:
        data = f.read(1000)
    
    end_header = b"end_header"
    i = 0
    while True:
        if data[i:i+len(end_header)] == end_header:
            break
        i += 1

    header = data[0:i].decode()
    header_done = i+len(end_header)+1
    header_lines = header.split("\n")

    fmat_ = header_lines[1]
    if "binary" in fmat_:
        fmat = "binary"
        if "big_endian" in fmat_:
            endness = "big endian"
        else:
            endness = "little endian"
    else:
        fmat = "ascii"

    if fmat == "binary":
        open_as = "rb"
    else:
        open_as = "r"
    
    with open(ply_path, open_as) as f:
        data = f.read()

    if fmat == "binary":
        pass
    else:
        header_done = data.index("end_header") + len(end_header) + 1

    vertex_properties = []
    face_properties = []

    current = None

    for i in header_lines:
        s = i.split(" ")
        if len(s) < 2:
            continue
        if "vertex" in s[1]:
            current = vertex_info = s
        elif "face" in s[1]:
            current = face_info = s
        else:
            if "property" not in i:
                continue
            if "vertex" in current:
                vertex_properties.append(i)
            elif "face" in current:
                face_properties.append(i)

    n_faces = int(face_info[2])
    n_verts = int(vertex_info[2])

    if fmat == "ascii":
        i = header_done
        count = 0
        while count < n_verts:
            if data[i] == "\n":
                count += 1
            i += 1

        vert_end = i
    
        vertex_excerpt = StringIO(data[header_done:vert_end])
        vertices = np.loadtxt(vertex_excerpt, dtype=np.float32)

        faces_excerpt = StringIO(data[vert_end:])
        faces = np.loadtxt(faces_excerpt, dtype=np.uint32)
        
    elif fmat == "binary":
       # print(header)
        if endness == "big endian":
            pat_verts = ">fff"
            pat_faces = ">Biii"
        else:
            pat_verts = "<fff"
            pat_faces = "<Biii"
        
        n_floats = n_verts * 3
        bytes_to_read = n_floats * 4
        vert_end = header_done + bytes_to_read
        vdata = data[header_done:vert_end]
        
        vertices = np.zeros((n_verts,3), dtype=np.float32)
        b = 0
        i = 0
        import struct
        while b < bytes_to_read:
            up = struct.unpack(pat_verts, vdata[b:b+12])
            vertices[i] = up
            i += 1
            b += 12
            del up
        
        #vertex_excerpt = BytesIO(ndata)
        #vertices = np.fromfile(vertex_excerpt, dtype=">f")
            
        n_ints = n_faces * 3
        n_uchars = n_faces
        bytes_to_read = n_ints * 4 + n_uchars * 1
        fdata = data[vert_end:]
        
        faces = np.zeros((n_faces,4), dtype=np.uint32)
        b = 0
        i = 0
        read_size = 4 * 3 + 1
        while b < bytes_to_read:
            up = struct.unpack(pat_faces, fdata[b:b+read_size])
            faces[i,0] = up[0]
            faces[i,1:] = up[1:]
            i += 1
            b += read_size
            del up

        #faces_excerpt = BytesIO(data[vert_end:])
        #faces = np.fromfile(faces_excerpt, dtype=np.uint32)

    #tri = np.zeros((n_verts*3,), dtype=np.float32)
    #tri[:n_verts] = vertices[faces[::,1]]
    #tri[n_verts:n_verts * 2] = vertices[faces[::,2]]
    #tri[n_verts * 2:] = vertices[faces[::,3]]
    # print(faces[::,1:])

    if faces[0][0] == 3:
        tris = np.zeros((faces.shape[0],3,3), dtype=np.float32)
        tris[::,0] = vertices[faces[::,1],:3]
        tris[::,1] = vertices[faces[::,2],:3]
        tris[::,2] = vertices[faces[::,3],:3]
    elif faces[0][0] == 4:
        pass

    return tris 

import myformat
def load_onlyverts_myformat(path_to_conv):
    S = path_to_conv.split("/")
    folder, name = "/".join(S[:-1])+"/", S[-1]
    
    mf = myformat.myformat()
    mf.load(folder, name)

    return mf.verts
