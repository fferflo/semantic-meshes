#!/usr/bin/env python3

import argparse, os, imageio, sys, semantic_meshes, shutil, subprocess, sqlite3
import numpy as np
from tqdm import tqdm
from columnar import columnar
import scipy.spatial.transform

parser = argparse.ArgumentParser(description="Create colmap reconstructions of the scannet dataset")
parser.add_argument("--scannet", type=str, required=True, help="Path to scannet directory")
parser.add_argument("--frames_step", type=int, default=1, help="Only use every n-th frame from a scene's frames")
parser.add_argument("--temp", type=str, required=True, help="Path to temporary directory where scene images are stored")
parser.add_argument("--once", action="store_true", help="Flag indicating that only the single next scene should be reconstructed")
args = parser.parse_args()

class RunException(BaseException):
    def __init__(self, message, code):
        self.message = message
        self.code = code

def run(command, log=None, echo=True):
    print("> " + command)
    if not log is None:
        log.write("> " + command + "\n")
    process = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE)
    result = ""
    reader = process.stdout
    while True:
        c = reader.read(1).decode("utf-8")
        if not c:
            break
        result += c
        if echo:
            print(c, end="")
        if not log is None:
            log.write(c)
    process.communicate()
    if process.returncode != 0:
        msg = result + "\nFailed to run " + command + ". Got return code " + str(process.returncode)
        raise RunException(msg, process.returncode)
    return result

# Search for scenes
scenes = sorted([os.path.join(args.scannet, "scans", f) for f in os.listdir(os.path.join(args.scannet, "scans")) if f.startswith("scene")])
print(f"Found {len(scenes)} scenes in {args.scannet}")

one_done = False
for scene in scenes:
    workspace = os.path.join(scene, "colmap")
    if (one_done and args.once) or os.path.isdir(workspace):
        continue
    one_done = True

    # Load scannet scene
    print(scene)
    with open(os.path.join("/home/ferflo/scannet_colmap_status.txt"), "w") as f:
        f.write(scene)
    name = os.path.basename(scene)
    sens = semantic_meshes.data2.SensFile(os.path.join(scene, name + ".sens"), max_frames=None)
    resolution = np.array([sens.color_height, sens.color_width])

    sens.frames = sens.frames

    # Run colmap
    if os.path.isdir(args.temp):
        print(f"Temporary directory {args.temp} already exists")
        sys.exit(-1)
    print(f"Saving images to {args.temp}")
    os.makedirs(args.temp)
    for frame_index, frame in tqdm(list(enumerate(sens.frames[::args.frames_step]))):
        color = frame.decompress_color_jpeg()
        shape = color.shape
        imageio.imwrite(os.path.join(args.temp, f"frame{frame_index:05}.png"), color)

    os.makedirs(workspace)
    database = os.path.join(workspace, "database.db")
    fused = os.path.join(workspace, "fused.ply")
    mesh = os.path.join(workspace, "mesh.ply")

    with open(os.path.join(workspace, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {shape[1]} {shape[0]} {sens.intrinsic_color[0][0]} {sens.intrinsic_color[1][1]} {sens.intrinsic_color[0][2]} {sens.intrinsic_color[1][2]}")
    with open(os.path.join(workspace, "points3D.txt"), "w") as f:
        pass

    try:
        run(f"colmap feature_extractor --database_path {database} --image_path {args.temp}")
        run(f"colmap exhaustive_matcher --database_path {database}")
    except RunException as e:
        shutil.rmtree(workspace)
        shutil.rmtree(args.temp)
        continue

    connection = sqlite3.connect(os.path.join(workspace, "database.db"))
    c = connection.cursor()
    c.execute("SELECT image_id, name FROM images")
    image_id_to_name = sorted(c.fetchall(), key=lambda x: x[1])
    name_to_pose = {f"frame{frame_index:05}.png": frame.camera_to_world for frame_index, frame in enumerate(sens.frames[::args.frames_step])}
    with open(os.path.join(workspace, "images.txt"), "w") as f:
        for image_id, name in image_id_to_name:
            camera_to_world = np.linalg.inv(name_to_pose[name])
            q = scipy.spatial.transform.Rotation.from_matrix(camera_to_world[:3, :3]).as_quat()
            t = camera_to_world[:3, 3]
            f.write(f"{image_id}, {q[3]}, {q[0]}, {q[1]}, {q[2]}, {t[0]}, {t[1]}, {t[2]}, 1, {name}\n\n")
    connection.close()

    try:
        run(f"colmap point_triangulator --database_path {database} --image_path {args.temp} --input_path {workspace} --output_path {workspace}")
        run(f"colmap image_undistorter --image_path {args.temp} --input_path {workspace} --output_path {workspace}")
        run(f"colmap patch_match_stereo --workspace_path {workspace}")
        run(f"colmap stereo_fusion --workspace_path {workspace} --output_path {fused}")
        run(f"colmap delaunay_mesher --input_path {workspace} --output_path {mesh} --DelaunayMeshing.quality_regularization 5. --DelaunayMeshing.max_proj_dist 10")
    except RunException as e:
        shutil.rmtree(workspace)
        shutil.rmtree(args.temp)
        continue

    shutil.rmtree(args.temp)
    shutil.rmtree(os.path.join(workspace, "images"))
    shutil.rmtree(os.path.join(workspace, "stereo"))
