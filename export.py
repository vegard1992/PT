def export_colored_ply(name, tris, colors):
    nfaces = len(tris)
    nvertices = nfaces * 3
    header = """ply
format ascii 1.0
comment object : triangular faces
element vertex {0}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {1}
property list uchar int vertex_index
end_header
""".format(nvertices, nfaces)
    text = header
    i = 0
    for t in tris:
        c = colors[i]
        for v in t:
            text += "{0} {1} {2}".format(*v)
            text += " {0} {1} {2}".format(*c)
            text += "\n"
        i += 1
        
    i = 0
    for f in range(nfaces):
        text += "3 {0} {1} {2}".format(i, i+1, i+2)
        text += "\n"
        i += 3

    with open(name+".ply", "w+") as f:
        f.write(text)

def export_tree(name, split):
    tris = []
    colors = []
    
    for n in get_split_faces(split):
        if n.w < 0 or n.h < 0 or n.l < 0:
            print(n.x, n.y, n.z)
            print(n.w, n.h, n.l)
        color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        b = n.get_faces()
        for k in b.keys():
            f = b[k]
            t1 = f[0], f[1], f[2]
            t2 = f[2], f[3], f[1]
            tris.append(t1)
            tris.append(t2)
            colors.append(color)
            colors.append(color)

    export_colored_ply(name, tris, colors)

def export_verts(name, verts):
    tris = []
    colors = []
    for i in range(int(len(verts)/3)):
        sliver = verts[i*3:i*3+3,:3]
        tris.append(sliver)
        color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        colors.append(color)

    export_colored_ply(name, tris, colors)
