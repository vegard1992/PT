from math import acos
import numpy as np
from export import export_colored_ply
import random


def sub(a, b):
    return a[0]-b[0], a[1]-b[1], a[2]-b[2]

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    Vx, Vy, Vz = a
    Wx, Wy, Wz = b
    Nx = Vy*Wz - Vz*Wy
    Ny = Vz*Wx - Vx*Wz
    Nz = Vx*Wy - Vy*Wx
    return Nx, Ny, Nz


def ray_plane_ipoint(RO, RD, A, B, C, D):
    X0, Y0, Z0 = RO
    Xd, Yd, Zd = RD
    t = -(A*X0 + B*Y0 + C*Z0 + D) / (A*Xd + B*Yd + C*Zd)
    x, y, z = X0+Xd*t, Y0+Yd*t, Z0+Zd*t
    return np.array((x, y, z, 0), dtype=np.float32)
    

def get_extremes(tris):
    e = 0.000001
    mx, my, mz = None, None, None
    Mx, My, Mz = None, None, None
    for t in tris:
        for p in t:
            x, y, z = p
            if mx == None or x < mx:
                mx = x
            if my == None or y < my:
                my = y
            if mz == None or z < mz:
                mz = z
            if Mx == None or x > Mx:
                Mx = x
            if My == None or y > My:
                My = y
            if Mz == None or z > Mz:
                Mz = z

    return (mx, my, mz), (Mx, My, Mz)

class SSplit:
    def __init__(self):
        self.bucket = None

        self.subdivided = False

        self.child1 = None
        self.child2 = None

        self.minv = 0, 0, 0
        self.maxv = 0, 0, 0

        self.plane = 0, 0, 0, 0

        self.depth = 0

#test_splitting()
def split_it(leaf, S):
    leaf.child1 = SSplit()
    leaf.child2 = SSplit()

    d1 = np.array((1, 0, 0), dtype=np.float32) # pointing right
    d2 = np.array((0, 1, 0), dtype=np.float32) # pointing forward
    d3 = np.array((0, 0, 1), dtype=np.float32) # pointing upward
    d = d1, d2, d3
    n = random.choice(d)
    A, B, C = n
    D = -dot(n, S[:3])

    leaf.plane = np.array((A, B, C, D), dtype=np.float32)

    i_n = 1 - n
    divide = leaf.maxv * i_n + S[:3] * n
    
    leaf.child1.minv = leaf.minv
    leaf.child1.maxv = divide
    leaf.child2.minv = divide
    leaf.child2.maxv = leaf.maxv
    leaf.child1.depth = leaf.depth+1
    leaf.child2.depth = leaf.depth+1
    #print("p", leaf.minv, leaf.maxv)
    #print("c1", leaf.child1.minv, leaf.child1.maxv)
    #print("c2", leaf.child2.minv, leaf.child2.maxv)
    leaf.subdivided = True
    
def get_leaves(leaf):
    if leaf.subdivided == False:
        return [leaf]
    else:
        leaves = []
        leaves += get_leaves(leaf.child1)
        leaves += get_leaves(leaf.child2)
        
    return leaves

def construct_polygon(tris):
    pass

def flatten_tris(tris):
    vertices = np.zeros((len(tris)*3, 4), dtype=np.float32)
    vertices[:len(tris)*3][::3,:3] = tris[::,0]
    vertices[:len(tris)*3][1::3,:3] = tris[::,1]
    vertices[:len(tris)*3][2::3,:3] = tris[::,2]

    return vertices

def point_plane_distance(p, plane):
    return dot(p[:3], plane[:3]) + plane[3]

def split_triangle2(vertices, plane):

    epsilon = 0.000001
    g1_ind = g2_ind = g3_ind = 0
    
    tindx = 0
    cid = 0
    out_vertices = list([0, 0, 0] for i in range(9))
    points_in_plane = 0
    g1 = list([0, 0, 0] for i in range(3))
    g2 = list([0, 0, 0] for i in range(3))
    g3 = list([0, 0, 0] for i in range(3))
    for i in range(3):
        p = vertices[tindx+i];
        distance = point_plane_distance(p, plane);
        if(distance < -epsilon):
            g2[g2_ind] = p;
            g2_ind += 1;
        
        elif(distance > epsilon):
            g1[g1_ind] = p;
            g1_ind += 1;
        else:
            points_in_plane += 1;
            g3[g3_ind] = p;
            g3_ind += 1;
            
    t0 = vertices[tindx];
    t1 = vertices[tindx+1];
    t2 = vertices[tindx+2];

    into_three = 3
    into_two = 2
    dont_split = 1
    
    if( (((g1_ind == 1) and (g2_ind == 2)) or ((g1_ind == 2) and (g2_ind == 1)))):
        split_type = into_three;
    elif(points_in_plane == 1):
        if((g1_ind == g2_ind == 1)):
            split_type = into_two;
        else:
            split_type = dont_split;
    else:
        split_type = dont_split;

    #print(split_type)

    vert_type = split_type
    R = 0, 0
    if(vert_type == 1):
        out_vertices[cid] = t0;
        out_vertices[cid+1] = t1;
        out_vertices[cid+2] = t2;
        R = 0, 3
        #//out_tri_indexes[last_tri_indexes_index] = rel_child_ind;
    
    elif(vert_type == 2) and 0:
        #// now split irregular
        tsi1 = g1[0];
        tsi2 = g2[0];
        osi = g3[0];
        RO = tsi1;
        RD = sub(tsi2, tsi1);
        inter1 = ray_plane_ipoint(RO[:3], RD[:3], *plane);

        t0 = osi; t1 = inter1; t2 = tsi1;
        out_vertices[cid] = t0;
        out_vertices[cid+1] = t1;
        out_vertices[cid+2] = t2;

        t0 = osi; t1 = inter1; t2 = tsi2;
        out_vertices[cid+3] = t0;
        out_vertices[cid+1+3] = t1;
        out_vertices[cid+2+3] = t2;

        R = 0, 6
    
    elif (vert_type == 3):
        if(g1_ind == 2):
            tsi1 = g1[0];
            tsi2 = g1[1];
            osi = g2[0];
        
        elif(g2_ind == 2):
            tsi1 = g2[0];
            tsi2 = g2[1];
            osi = g1[0];
        
        RO = osi;
        RD = tsi1-osi;#sub(tsi1, osi);
        inter1 = ray_plane_ipoint(RO[:3], RD[:3], *plane);
        RO = osi;
        RD = tsi2-osi;#sub(tsi2, osi);
        inter2 = ray_plane_ipoint(RO[:3], RD[:3], *plane);

        t0 = osi; t1 = inter1; t2 = inter2;
        out_vertices[cid] = t0;
        out_vertices[cid+1] = t1;
        out_vertices[cid+2] = t2;

        a = inter2-inter1;#sub(inter2, inter1);
        b = tsi1-inter1;#sub(tsi1, inter1);
        c = tsi2-inter1;#sub(tsi2, inter1);
        angle1 = (dot(a[:3], b[:3]) / ((dot(a[:3], a[:3])**0.5) * (dot(b[:3], b[:3])**0.5)));
        angle2 = (dot(a[:3], c[:3]) / ((dot(a[:3], a[:3])**0.5) * (dot(c[:3], c[:3])**0.5)));
        if(acos(angle1) < acos(angle2)):
            third = tsi1;
            fourth = tsi2;
        
        else:
            third = tsi2;
            fourth = tsi1;
        

        t0 = inter1; t1 = inter2; t2 = third;
        out_vertices[cid+3] = t0;
        out_vertices[cid+1+3] = t1;
        out_vertices[cid+2+3] = t2;

        t0 = third; t1 = fourth; t2 = inter1;
        out_vertices[cid+6] = t0;
        out_vertices[cid+1+6] = t1;
        out_vertices[cid+2+6] = t2;

        R = 0, 9

    OUT = out_vertices[R[0]:R[1]]

    return OUT

def export_verts(name, verts):
    tris = []
    colors = []
    for i in range(int(len(verts)/3)):
        sliver = verts[i*3:i*3+3,:3]
        tris.append(sliver)
        color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        colors.append(color)

    export_colored_ply(name, tris, colors)
    
def split_method_2():
    print("LOADING SCENE..")
    tri1 = (1, 1, 70), (70, 1, 70), (70, 70, 70)
    tri2 = (20, 20, 80), (60, 60, 60), (20, 60, 80)
    tri3 = (1, 1, 1), (99, 99, 99), (1, 1, 99)

    tri4 = (1, 1, 1), (2, 2, 2), (1, 2, 1)
    tri5 = (97, 97, 98), (98, 97, 98), (96, 96, 97)

    tris = np.array((tri1, tri2, tri3), dtype=np.float32)
    #tris = tri4, tri5
    from convert import ply2scene

    #tris = ply2scene("scenes/bunny/bun_zipper_res4.ply")
    tris = ply2scene("scenes/beethoven.ply")
    vertices = flatten_tris(tris)
    
    n_verts = len(vertices)
    n_tris = len(tris)
    
    print(len(tris))
    print("..DONE")

    init = SSplit()
    ex = get_extremes(tris)
    init.minv = ex[0][:3]
    init.maxv = ex[1][:3]
    
    leaves = get_leaves(init)
    for leaf in leaves:
        S = np.array((np.sum(vertices[::,0]), \
                      np.sum(vertices[::,1]), \
                      np.sum(vertices[::,2]), \
                      0), \
                     dtype=np.float32)
        
        S /= n_verts
        split_it(leaf, S)

    print(init.plane)
        
    new_verts = []
    for i in range(n_tris):
        indx = i * 3
        sliver = vertices[indx:indx+3]
        new_verts += split_triangle2(sliver, init.plane)

    print(new_verts[:10])
    for i in new_verts:
        if len(i) != 4:
            print(len(i))
            print("EXCEPTION")
            print(i)
            raise Exception()
    new_verts = np.array(new_verts)
    export_verts("py_split_verts", new_verts)

split_method_2()

# so,
# if everything was divided into polygons
# i could get a plane along each face
# and do delaunay along that plane for every point that lies in it
# thus sorting out the "bad" triangles

# alternatively just start with one big polygon
# split that one


