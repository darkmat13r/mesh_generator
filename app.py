from flask import Flask,  request
import open3d as o3d
import numpy as np
import os
from flask import send_file
app = Flask(__name__)

@app.route('/generate', methods = ['POST'])
def generate():  # put application's code here
    f = request.files["point_cloud"]
    res = './'
    print(res)
    print(f.filename)
    filePath = res  + f.filename
    f.save(filePath)

    pcd = read(filePath)
    pcd = down_sample(pcd)
    recalculate_normals(pcd)
    generatedFile = generate_mesh(pcd, filePath)
    print("Print a normal vector of the 0th point")
    print(pcd.normals[0])
    os.remove(filePath)
    return send_file(generatedFile, as_attachment = True)

@app.route('/', methods = ['POST'])
def hello_world():  # put application's code here
    pcd = read("./scan_2022_10_25_09_45_37.obj")
    pcd = down_sample(pcd)
    recalculate_normals(pcd)
    generate_mesh(pcd)
    print("Print a normal vector of the 0th point")
    print(pcd.normals[0])
    return "completed"

def generate_mesh(pcd, filePath):
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    print('visualize densities')
    densities = np.asarray(densities)

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)

    opFile = filePath.replace(".ply", ".obj")
    o3d.io.write_triangle_mesh(opFile, mesh)
    return opFile

def read(file):
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(file)
    return pcd

def visualize(pcd):
    print("Load a ply point cloud, print it, and render it")
    print(pcd)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
def down_sample(pcd):
    return pcd.voxel_down_sample(voxel_size=0.001)


def recalculate_normals(pcd):
    print("Recompute the normal of the downsampled point cloud")
    return pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


if __name__ == '__main__':
    app.run()
