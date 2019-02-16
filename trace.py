import time
import math
import random
import sys


import pyopencl as cl
mf = cl.mem_flags
import numpy as np
from imageio import imwrite


from load import load_onlyverts_myformat
from export import export_colored_ply, export_tree, export_verts




def cl_init():
    platforms = cl.get_platforms()
    for p in platforms:
        if p.get_devices()[0].type == 4:
            platform = p
            break
    context = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platform)])

    queue = cl.CommandQueue(context)

    return platform, context, queue

def save_image(output, key, folder):
    imwrite(folder+'outfile_{0}_{1}.jpg'.format(*key), output)

def dump_buf(output):
    with open("output.txt", "w+") as f:
        f.write("\n".join(map(lambda x: str(list(map(lambda y: str(list(y)), x))), output[::4, ::4])))

def grab_kernel(kernel_path):
    with open(kernel_path, "r") as f:
        kernel = f.read()
    return kernel

def main():
    w, h = 900, 1600
    platform, context, queue = cl_init()

    kernel = grab_kernel("kernel.cl")
    program = cl.Program(context, kernel).build()

    loop(context, queue, program, w, h)

def get_extremes_min(ctx, program, queue, vertices):
    size = 2**int(math.log(vertices.shape[0], 2) + 1)
    shape = [size] + list(vertices.shape[1:])
    thingy = np.zeros(shape, dtype=vertices.dtype)
    thingy[:vertices.shape[0]] = vertices
    thingy[vertices.shape[0]:] = vertices[0]

    groupindx = 0
    groupsize = thingy.shape[0]

    used = 0

    while thingy.shape[0] > 1:
        groupsize = int(groupsize/2)
        if groupsize == 0:
            groupsize = 1

        thingy2 = np.copy(thingy[::2])

        q_thingy = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy)
        q_thingy2 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy2)
        
        kernelargs = (q_thingy, q_thingy2)
        program.extremes_min(queue, (groupsize,), None, *(kernelargs))
        cl.enqueue_copy(queue, thingy2, q_thingy2).wait()
        
        thingy = np.copy(thingy2)
        

    minv = thingy[0]

    return minv

def get_extremes_max(ctx, program, queue, vertices):
    size = 2**int(math.log(vertices.shape[0], 2) + 1)
    shape = [size] + list(vertices.shape[1:])
    thingy = np.zeros(shape, dtype=vertices.dtype)
    thingy[:vertices.shape[0]] = vertices
    thingy[vertices.shape[0]:] = vertices[0]

    groupindx = 0
    groupsize = thingy.shape[0]

    used = 0

    while thingy.shape[0] > 1:
        groupsize = int(groupsize/2)
        if groupsize == 0:
            groupsize = 1

        thingy2 = np.copy(thingy[::2])

        q_thingy = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy)
        q_thingy2 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy2)
        
        kernelargs = (q_thingy, q_thingy2)
        program.extremes_max(queue, (groupsize,), None, *(kernelargs))
        cl.enqueue_copy(queue, thingy2, q_thingy2).wait()
        
        thingy = np.copy(thingy2)
        

    maxv = thingy[0]

    return maxv

def get_extremes(context, program, queue, vertices):
    minv = get_extremes_min(context, program, queue, vertices)
    maxv = get_extremes_max(context, program, queue, vertices)
    return minv, maxv

def get_sum_verts(ctx, program, queue, vertices):    
    size = 2**int(math.log(vertices.shape[0], 2) + 1)
    shape = [size] + list(vertices.shape[1:])
    thingy = np.zeros(shape, dtype=vertices.dtype)
    thingy[:vertices.shape[0]] = vertices

    groupindx = 0
    groupsize = thingy.shape[0]

    used = 0

    while thingy.shape[0] > 1:
        groupsize = int(groupsize/2)
        if groupsize == 0:
            groupsize = 1

        thingy2 = np.copy(thingy[::2])

        q_thingy = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy)
        q_thingy2 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy2)
        
        kernelargs = (q_thingy, q_thingy2)
        program.sum_verts(queue, (groupsize,), None, *(kernelargs))
        cl.enqueue_copy(queue, thingy2, q_thingy2).wait()
        
        thingy = np.copy(thingy2)
        

    sumv = thingy[0]

    return sumv

def get_sum(ctx, program, queue, arr):
    size = 2**int(math.log(arr.shape[0], 2) + 1)
    shape = [size] + list(arr.shape[1:])
    thingy = np.zeros(shape, dtype=arr.dtype)
    thingy[:arr.shape[0]] = arr

    groupindx = 0
    groupsize = thingy.shape[0]

    used = 0

    while thingy.shape[0] > 1:
        groupsize = int(groupsize/2)
        if groupsize == 0:
            groupsize = 1

        thingy2 = np.copy(thingy[::2])

        q_thingy = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy)
        q_thingy2 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy2)
        
        kernelargs = (q_thingy, q_thingy2)
        program.sum(queue, (groupsize,), None, *(kernelargs))
        cl.enqueue_copy(queue, thingy2, q_thingy2).wait()
        
        thingy = np.copy(thingy2)

    sumv = thingy[0]

    return sumv

def prefix_sums(ctx, program, queue, arr):

    alen = arr.shape[0]
    divides = int(math.log(alen, 2)) + 1
    size = 2**divides
    new_arr = np.zeros((size,), dtype=np.int32)
    new_arr[:alen] = arr
    sgeo = lambda a1, r, n: int((a1 / (1-r)) * (1-r**(n+1)))
    
    sums_shape = (sgeo(1, 2, divides), )
    sums = np.zeros(sums_shape, dtype=np.int32)
    sums[:new_arr.shape[0]] = new_arr

    q_sums = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=sums)

    thingy = new_arr

    groupindx = 0
    groupsize = thingy.shape[0]

    used = 0

    index = 0
    index_ = 0

    while thingy.shape[0] > 1:
        index += groupsize

        groupsize = int(groupsize/2)
        if groupsize == 0:
            groupsize = 1

        thingy2 = np.copy(thingy[::2])
        q_thingy = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy)
        q_thingy2 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=thingy2)
        
        kernelargs = (q_thingy, q_thingy2, q_sums, np.int32(index))
        program.prefix_sums_sum(queue, (groupsize,), None, *(kernelargs))
        cl.enqueue_copy(queue, thingy2, q_thingy2).wait()

        thingy = np.copy(thingy2)

    cl.enqueue_copy(queue, sums, q_sums).wait()
    #print(np.copy(sums[::-1]))
    tree = np.zeros((sums_shape[0]*3,), dtype=np.int32)
    q_tree = cl.Buffer(ctx, mf.READ_WRITE, tree.nbytes)
   
    groupsize = sums_shape[0]

    q_sums_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.copy(sums[::-1]))
    kernelargs = (q_sums_in, q_tree)
    program.prefix_sums_tree(queue, (groupsize,), None, *(kernelargs))
    cl.enqueue_copy(queue, tree, q_tree).wait()

    another_arr = np.zeros((alen,), dtype=np.int32)
    q_another_arr = cl.Buffer(ctx, mf.WRITE_ONLY, another_arr.nbytes)

    groupsize = alen
    kernelargs = (q_tree, q_another_arr, np.int32(divides), np.int32(3))
    program.prefix_sum(queue, (groupsize,), None, *(kernelargs))
    cl.enqueue_copy(queue, another_arr, q_another_arr)

    result = another_arr[:alen]

    return result

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

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


def serialize_tree(node):
    serialized = []
    serialized_planes = []
    serialized_m = []
    serialized_M = []

    cindex = 1
    index = 0
    entry_len = 5
    children = [node]
    while True:
        node = children.pop(0)
        if node.child1 != None:
            children.append(node.child1)
        if node.child2 != None:
            children.append(node.child2)
            
        depth = node.depth

        serialized.append((cindex) * entry_len)
        cindex += 1
        serialized.append((cindex) * entry_len)
        cindex += 1

        serialized.append(index)
        serialized.append(index)
        serialized.append(index)
        
        plane = node.plane
        serialized_planes.append(plane)

        mv = node.minv
        serialized_m.append(mv)
        Mv = node.maxv
        serialized_M.append(Mv)

        index += 1

        if len(children) == 0:
            break
    
    return np.array(serialized, dtype=np.int32), np.array(serialized_planes, dtype=np.float32), \
           np.array(serialized_m, dtype=np.float32), np.array(serialized_M, dtype=np.float32)

def get_split_faces(split):
    faces = []
    leaves = get_leaves(split)
    for leaf in leaves:
        n = Node()
        n.x, n.y, n.z = leaf.minv
        n.w, n.h, n.l = leaf.maxv-leaf.minv
        faces.append(n)

    return faces

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

def build_bvh(bucket_size, vertices, ex, cl_handle):
    ctx, program, queue = cl_handle

    n_verts = vertices.shape[0]
    n_tris = int(n_verts / 3)

    #divides = int(math.log(n_tris/bucket_size, 2)+1)
    divides = 5

    init = SSplit()
    init.bucket = vertices
    init.minv = ex[0][:3]
    init.maxv = ex[1][:3]

    tri_indexes_entry_len = 1
    tri_ind = np.zeros((n_tris,), dtype=np.int32)

    bufmein = lambda x: cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    bufmeout1 = lambda x: cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    bufmeout2 = lambda x: cl.Buffer(ctx, mf.WRITE_ONLY, x.nbytes)
    
    for i in range(divides): # on the 2nd iteration serialize_tree gets much bigger
        
        leaves = get_leaves(init)
        for leaf in leaves:
            S = get_sum_verts(ctx, program, queue, leaf.bucket)
            S /= n_verts

            split_it(leaf, S)
        
        q_vertices = bufmein(vertices)
        serialized = serialize_tree(init)
        Bufs_ = list(bufmein(Buf) for Buf in serialized)
        q_tree_ind, q_planes, q_mv, q_Mv = Bufs_

        q_tri_ind = bufmein(tri_ind)

        out_v_type = np.empty((n_tris,), dtype=np.int32)
        q_out_v_type = bufmeout2(out_v_type)

        out_c_belong = np.empty((n_tris,), dtype=np.int32)
        q_out_c_belong = bufmeout2(out_c_belong)

        kernelargs = (q_vertices, q_planes, q_mv, q_Mv, q_tree_ind, q_tri_ind, \
                      q_out_v_type, q_out_c_belong, \
                      np.int32(tri_indexes_entry_len))
        program.determine_splits(queue, (n_tris,), None, *(kernelargs))
        cl.enqueue_copy(queue, out_v_type, q_out_v_type).wait()
        cl.enqueue_copy(queue, out_c_belong, q_out_c_belong).wait()

        tri_indexes_entry_len += 1

        cum_v_type = prefix_sums(ctx, program, queue, np.copy(out_v_type))

        q_cum_v_type = bufmein(cum_v_type)
        c_belong = bufmein(out_c_belong)

        new_n_tris = get_sum(ctx, program, queue, out_v_type)

        out_v = np.zeros((new_n_tris*3, 4), dtype=np.float32)
        q_out_v = bufmeout2(out_v)

        out_tindxs = np.zeros((new_n_tris*tri_indexes_entry_len,), dtype=np.int32)
        q_out_tindxs = bufmeout2(out_tindxs)

        kernelargs = (q_vertices, q_planes, q_mv, q_Mv, q_tree_ind, q_tri_ind, \
                      q_out_v_type, q_cum_v_type, q_out_c_belong, q_out_v, q_out_tindxs, \
                      np.int32(tri_indexes_entry_len-1))
        program.split_triangles(queue, (n_tris,), None, *(kernelargs))
        cl.enqueue_copy(queue, out_v, q_out_v).wait()
        cl.enqueue_copy(queue, out_tindxs, q_out_tindxs).wait()

        vertices = out_v
        tri_ind = out_tindxs

        n_tris = new_n_tris

        break
    
    #export_verts("split_tris", vertices)

    # how will my tree look?
    # vertices ofc
    # and i need the actual serialized leaves -
    # child1, child2, plane, mv, Mv, contained
    # and then contained has two entries; start and end
    # which point to another array which has
    # the indexes of every triangle that is in the box
    # with index array to what the leaves contain!

    return vertices, 



def loop(context, queue, program, w, h):
    print("loading model..")
    #vertices = load_onlyverts_myformat("scenes/skulls/myformat/damaliscus")
    vertices = load_onlyverts_myformat("scenes/mengersponge/myformat/mengersponge_i3")
    #vertices = load_onlyverts_myformat("scenes/mengersponge/myformat/mengersponge_i3")
    #vertices = load_onlyverts_myformat("scenes/happy/myformat/happy_vrip_res3")
    #vertices = load_onlyverts_myformat("scenes/dragon/myformat/dragon_vrip_res4")
    n_tris = int(len(vertices) / 3)
    print("tris:", n_tris)
    print("..loaded")
    
    print("computing extremes..")
    E0, E1 = get_extremes(context, program, queue, vertices)
    print("..done")

    print("scaling..")
    w_, h_, l_ = E1[0]-E0[0], E1[1]-E0[1], E1[2]-E0[2]
    s1, s2, s3 = 2/w_, 2/h_, 2/l_
    s = min((s1, s2, s3))
    vertices = vertices * s - 1.5
    E0 = E0 * s - 1.5
    E1 = E1 * s - 1.5
    print("..done")


    print("building bvh..")
    bvh = build_bvh(250, vertices, (E0, E1), (context, program, queue))
    print("..built")
    
    ctx = context
    shape = w, h
    
    q = vertices
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)

    output = np.empty((shape[0], shape[1], 3), dtype=np.float32)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    dt = 0.01
    t = time.clock()
    saved = False

    zx, zy = 8, 8
    zones = []
    zw, zh = int(w/zx), int(h/zy)
    for x in range(zx):
        for y in range(zy):
            zp = x * zw, y * zh
            zones.append(zp)

    rlen = w * h
    np_rand = np.random.random((rlen,)).astype(np.float32)
    q_random = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np_rand)
    
    while True:
        for bx, by in zones:
            #kernelargs = (q_opencl, output_opencl, np.int32(w), np.int32(h), np.int32(n_tris), np.int32(bx), np.int32(by))
            kernelargs = (q_opencl, q_random, output_opencl, np.int32(w), np.int32(h), np.int32(n_tris), np.int32(bx), np.int32(by), np.int32(rlen))
            program.path_trace_simple_noaccel(queue, (zw, zh), None, *(kernelargs))
            cl.enqueue_copy(queue, output, output_opencl).wait()
            #save_image(output, (bx, by), "out/")
            # NO ->
            T = time.clock() - t
            if int(T) % dt == 0 and 0:
                if not saved:
                    worked_output = output
                    #dump_buf(worked_output)
                    print("image saved,", "time elapsed", T)
                    saved = True
            else:
                saved = False
        save_image(output, (0, 0), "")
        return
    
main()

# finish splitting;
# remember to assign children
# and maybe group triangles in the same go

# add reading in binary data with numpy?

# so for making spacial splits;
# for now;
# choose a random direction
# and the average point
# and split

# for later;
# choose a random direction? or have some heuristic for choosing it
# then use a kind of binary search where u start at half
# and then check the halves of those halves
# counting the # of points on each side, each time
# when theyre close enough, youre done!
