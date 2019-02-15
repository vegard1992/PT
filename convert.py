from load import loadply

import myformat
from utilities import flatten_tris
def convert_to_myformat(path_to_file, path_to_conv):
    S = path_to_conv.split("/")
    folder, name = "/".join(S[:-1])+"/", S[-1]
    
    mf = myformat.myformat()
    mf.verts = flatten_tris(loadply(path_to_file))
    mf.save(folder, name)

convert_to_myformat("scenes/mengersponge/mengersponge_i3.ply", \
                    "scenes/mengersponge/myformat/mengersponge_i3")



#verts = load_onlyverts_myformat("scenes/lucy/myformat/lucy")
