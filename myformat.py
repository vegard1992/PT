import numpy as np
import os

class myformat:
    def __init__(self):
        self.verts = None
        self.mats = None

        self.format_name = "quick"
        self.sub_files = {}
        self.sub_files["mats"] = "m"
        self.sub_files["verts"] = "v"

    def save(self, folder, name):
        verts_path = folder+name+"."+self.format_name+"_"+self.sub_files["verts"]
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(verts_path, "w+b") as f:
            self.verts.tofile(f)

    def load(self, folder, name):
        verts_path = folder+name+"."+self.format_name+"_"+self.sub_files["verts"]
        with open(verts_path, "rb") as f:
            self.verts = np.fromfile(f, dtype=np.float32)
        self.verts = self.verts.reshape((int(len(self.verts)/4), 4))
