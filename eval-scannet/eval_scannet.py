#!/usr/bin/env python3

import argparse, os, imageio, sys, tfcv, semantic_meshes, yaml
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from collections import defaultdict
from distinctipy import distinctipy
from columnar import columnar
import tinypl as pl
import threading, queue

parser = argparse.ArgumentParser(description="Evaluate semantic-meshes on the scannet dataset")
parser.add_argument("--scannet", type=str, required=True, help="Path to scannet directory")
parser.add_argument("--images_equal_weight", type=float, default=0.5, help="Soft boolean flag such that 0.0 weights images equally and 1.0 weights pixels equally in the fusion step")
parser.add_argument("--frames_step", type=int, default=1, help="Only use every n-th frame from a scene's frames")
parser.add_argument("--aggregator", type=str, default="mul", help="The type of aggregator to use for fusing pixel annotations") # TODO: choice
parser.add_argument("--debug", type=str, default=None, help="A path where debugging images and plys are stored for the first scene")
parser.add_argument("--output", type=str, default=None, help="File/ directory where results will be stored")
parser.add_argument("--mesh", type=str, default="scannet", help="The type of mesh that should be used")
parser.add_argument("--simplify", type=str, default="1.0", help="The factor by which the mesh has been simplified")
parser.add_argument("--offset", type=int, default=0, help="Skipping the first number of scenes")
parser.add_argument("--num", type=int, default=-1, help="Number of scenes to use")
parser.add_argument("--cache", type=str, default=None, help="Cache directory where rendered images are stored and loaded for reuse")
parser.add_argument("--mode", type=str, default="triangles", help="Argument determining whether 'triangles' or 'texels' meshes should be used for the fusion")
parser.add_argument("--texel_resolution", type=float, default=0.1, help="Texel resolution parameter. Only valid with '--mode texels'")

args = parser.parse_args()
assert args.mode in ["triangles", "texels"]

dont_care_threshold = 0.9
classes_num = 40
class_to_color = np.asarray(distinctipy.get_colors(classes_num)) * 255.0

# Search for scenes
scenes = sorted([os.path.join(args.scannet, "scans", f) for f in os.listdir(os.path.join(args.scannet, "scans")) if f.startswith("scene")])
total_scenes = len(scenes)
if args.mesh == "scannet":
    if float(args.simplify) == 1.0:
        scenes = [(scene, os.path.join(scene, os.path.basename(scene) + "_vh_clean_2.labels.ply")) for scene in scenes]
    else:
        scenes = [(scene, os.path.join(scene, os.path.basename(scene) + f"_vh_clean_2.labels_simplified-{args.simplify}.ply")) for scene in scenes]
elif args.mesh == "colmap":
    scenes = [(scene, os.path.join(scene, "colmap", "mesh.ply")) for scene in scenes]
else:
    raise ValueError(f"Invalid mesh type {args.mesh}")
scenes = [(scene, input_mesh_file) for scene, input_mesh_file in scenes if os.path.isfile(input_mesh_file)]
mesh_scenes = len(scenes)
scenes = scenes[args.offset:]
if args.num < 0:
    args.num = len(scenes)
scenes = scenes[:args.num]
subset_scenes = len(scenes)
print(f"Found {total_scenes} total scenes in {args.scannet}, {mesh_scenes} scenes with mesh type {args.mesh}, using {subset_scenes} scenes for evaluation")

# Prepare results
result = {"metrics": {}, "params": {}}
result["params"]["images_equal_weight"] = args.images_equal_weight
result["params"]["frames_step"] = args.frames_step
result["params"]["mode"] = args.mode
result["params"]["aggregator"] = args.aggregator
result["params"]["mesh"] = args.mesh
result["params"]["num"] = args.num
result["params"]["offset"] = args.offset
result["params"]["simplify"] = args.simplify
if args.mode == "texels":
    result["params"]["texel_resolution"] = args.texel_resolution

if not args.output is None and os.path.isdir(args.output):
    for file in os.listdir(args.output):
        if file.endswith(".yaml"):
            file = os.path.join(args.output, file)
            with open(file, "r") as f:
                result2 = yaml.safe_load(f)
            if result2["params"] == result["params"]:
                print(f"Parametrization already exists in {file}")
                sys.exit(0)
if not args.debug is None:
    if not os.path.isdir(args.debug):
        os.makedirs(args.debug)

# Load pretrained model
model = tfcv.model.pretrained.tuinicr.esanet_resnet_v1b_34_nbt1d_nyuv2.create()
predictor = lambda x: model(x, training=False)
preprocess = tfcv.model.pretrained.tuinicr.esanet_resnet_v1b_34_nbt1d_nyuv2.preprocess

# Read label map
print("Creating label maps from scannet to nyu40...")
with open(os.path.join(args.scannet, "scannetv2-labels.combined.tsv"), "r") as f:
    lines = f.read().split("\n")
lines = [l.strip() for l in lines]
lines = [l for l in lines if len(l) > 0]
header = lines[0].split()
columns = defaultdict(list)
index = 0
for line in lines[1:]:
    values = line.split("\t")
    assert len(values) <= len(header)
    for name, value in zip(header[:len(values)], values):
        columns[name].append(value.strip())
scannet_to_nyu40 = {int(s): int(n) for s, n in zip(columns["id"], columns["nyu40id"])}
scannet_to_nyu40 = np.asarray([(scannet_to_nyu40[i] if i in scannet_to_nyu40 else 0) for i in range(np.amax(list(scannet_to_nyu40.keys())) + 1)])
assert np.all(0 <= scannet_to_nyu40)
scannet_to_nyu40 = scannet_to_nyu40 - 1
assert np.all(scannet_to_nyu40 < 40)

vertex_metrics = [
    tfcv.metric.Accuracy(classes_num=classes_num, dontcare_prediction="error"),
    tfcv.metric.MeanIoU(classes_num=classes_num, dontcare_prediction="error"),
    tfcv.metric.ConfusionMatrix(classes_num=classes_num, dontcare_prediction="error"),
]
image_metrics_network = [
    tfcv.metric.Accuracy(classes_num=classes_num, dontcare_prediction="forbidden"),
    tfcv.metric.MeanIoU(classes_num=classes_num, dontcare_prediction="forbidden"),
    tfcv.metric.ConfusionMatrix(classes_num=classes_num, dontcare_prediction="forbidden"),
]
image_metrics_fused = [
    tfcv.metric.Accuracy(classes_num=classes_num, dontcare_prediction="error"),
    tfcv.metric.MeanIoU(classes_num=classes_num, dontcare_prediction="error"),
    tfcv.metric.ConfusionMatrix(classes_num=classes_num, dontcare_prediction="error"),
]
for scene_index, (scene, input_mesh_file) in enumerate(scenes):
    # Load scannet scene
    name = os.path.basename(scene)
    sens = semantic_meshes.data2.SensFile(os.path.join(scene, name + ".sens"), max_frames=None)
    resolution = np.array([sens.color_height, sens.color_width])

    # Load camera parameters
    if not np.all(np.isclose(sens.extrinsic_color, np.eye(4))):
        print(f"Invalid color extrinsics: {sens.extrinsic_color}")
        sys.exit(-1)
    focal_lengths = np.asarray([sens.intrinsic_color[0, 0], sens.intrinsic_color[1, 1]])
    principal_point = np.asarray([sens.intrinsic_color[0, 2], sens.intrinsic_color[1, 2]])
    intrinsic_color = np.asarray(sens.intrinsic_color)
    intrinsic_color[0, 0] = 1
    intrinsic_color[1, 1] = 1
    intrinsic_color[0, 2] = 0
    intrinsic_color[1, 2] = 0
    if not np.all(np.isclose(intrinsic_color, np.eye(4))):
        print(f"Invalid color intrinsics: {sens.intrinsic_color}")
        sys.exit(-1)

    # Initialize semantic-meshes
    print("Initializing semantic-meshes...")
    mesh = semantic_meshes.data.Ply(input_mesh_file)
    if args.mode == "triangles":
        renderer = semantic_meshes.render.triangles(mesh)
    else: # args.mode == "texels"
        cameras = []
        for frame in sens.frames[::args.frames_step]:
            camera_to_world = np.linalg.inv(frame.camera_to_world)
            rotation = camera_to_world[:3, :3]
            translation = camera_to_world[:3, 3]
            cameras.append(semantic_meshes.data.Camera(rotation, translation, np.asarray([resolution[1], resolution[0]]), focal_lengths, principal_point))
        renderer = semantic_meshes.render.texels(mesh, cameras, args.texel_resolution)
    aggregator = semantic_meshes.fusion.MeshAggregator(primitives=renderer.getPrimitivesNum(), classes=classes_num, aggregator=args.aggregator, images_equal_weight=args.images_equal_weight)

    print(f"Loaded scene with {renderer.getPrimitivesNum()} mesh-primitives and {len(sens.frames)} frames")

    # Fuse predictions on mesh
    stream = iter(list(enumerate(sens.frames))[::args.frames_step])
    num = len(sens.frames[::args.frames_step])

    dir_lock = threading.Lock()
    @pl.unpack
    def load(index, frame):
        primitive_indices_cached = None
        primitive_indices_cache_file = None
        if not args.cache is None:
            mode = "triangles" if args.mode == "triangles" else f"texels-{args.texel_resolution}"
            simplify = f"simplify-{args.simplify}"
            primitive_indices_cache_dir = os.path.join(args.cache, name, f"primitive_indices-{mode}-{simplify}")
            with dir_lock:
                if not os.path.isdir(primitive_indices_cache_dir):
                    os.makedirs(primitive_indices_cache_dir)
            primitive_indices_cache_file = os.path.join(primitive_indices_cache_dir, f"frame-{index}.npz")

            if os.path.isfile(primitive_indices_cache_file):
                primitive_indices_cached = np.load(primitive_indices_cache_file)
                if not "data" in primitive_indices_cached:
                    primitive_indices_cached = None
                else:
                    primitive_indices_cached = primitive_indices_cached["data"]
        return index, frame, primitive_indices_cached, primitive_indices_cache_file
    stream = pl.map(load, stream)
    stream = pl.queued(stream, workers=4, maxsize=8)

    q = queue.Queue(maxsize=3)
    stream_out = iter((q.get() for _ in range(num)))

    @pl.unpack
    def aggregate(primitive_indices, pred_probs, image_index, color, gt_probs):
        aggregator.add(primitive_indices, pred_probs)
        if not args.debug is None and image_index % 100 == 0:
            imageio.imwrite(os.path.join(args.debug, f"{image_index}_color.png"), color)
            #imageio.imwrite(os.path.join(args.debug, f"{image_index}_depth.png"), tf.where(tf.math.is_finite(depth), depth, 0).numpy())
            imageio.imwrite(os.path.join(args.debug, f"{image_index}_gt.png"), tfcv.util.colorize(segmentation=gt_probs, image=color, class_to_color=class_to_color))
            imageio.imwrite(os.path.join(args.debug, f"{image_index}_pred.png"), tfcv.util.colorize(segmentation=tf.transpose(pred_probs, (1, 0, 2)), image=color, class_to_color=class_to_color))
    stream_out = pl.map(aggregate, stream_out)
    stream_out = pl.queued(stream_out, workers=1)

    for index, frame, primitive_indices_cached, primitive_indices_cache_file in tqdm(stream, total=num):
        # Render mesh
        if primitive_indices_cached is None:
            camera_to_world = np.linalg.inv(frame.camera_to_world)
            rotation = camera_to_world[:3, :3]
            translation = camera_to_world[:3, 3]
            camera = semantic_meshes.data.Camera(rotation, translation, np.asarray([resolution[1], resolution[0]]), focal_lengths, principal_point)
            primitive_indices, depth = renderer.render(camera)
            depth = tf.expand_dims(tf.transpose(tf.experimental.dlpack.from_dlpack(depth), (1, 0)), axis=-1) * 1000.0
            primitive_indices = tf.transpose(tf.experimental.dlpack.from_dlpack(primitive_indices), (1, 0))
        else:
            primitive_indices = primitive_indices_cached

        # Predict classes
        color = frame.decompress_color_jpeg()
        # if args.mesh == "scannet":
        depth = np.frombuffer(frame.decompress_depth_zlib(), dtype=np.uint16).reshape(sens.depth_height, sens.depth_width)
        depth = tf.cast(tf.expand_dims(depth, axis=-1), "float32")
        depth_preprocessed = tf.image.resize(depth, (480, 640), method="nearest")
        color_preprocessed = tf.image.resize(color, (480, 640), method="bilinear")
        color_preprocessed, depth_preprocessed = preprocess(color_preprocessed, depth_preprocessed)
        pred_probs = predictor([np.expand_dims(color_preprocessed, axis=0), np.expand_dims(np.expand_dims(depth_preprocessed, axis=0), axis=-1)])[0]
        pred_probs = tf.image.resize(pred_probs, resolution, method="bilinear")

        if not args.cache is None and args.mesh == "scannet":
            if primitive_indices_cached is None:
                np.savez_compressed(primitive_indices_cache_file, data=primitive_indices)

        # Update 2d metrics
        gt_probs = imageio.imread(os.path.join(scene, "label-filt", f"{index}.png"))
        gt_probs = tf.gather(scannet_to_nyu40, tf.cast(gt_probs, "int32"))
        gt_probs = tf.one_hot(gt_probs, depth=classes_num)
        for m in image_metrics_network:
            m.update_state(gt_probs, pred_probs)

        q.put((tf.transpose(primitive_indices, (1, 0)).numpy(), tf.transpose(pred_probs, (1, 0, 2)).numpy(), index, color, gt_probs))
    stream_out.stop()
    print("Computing primitive annotations...")
    pred_primitive_annotations = tf.convert_to_tensor(aggregator.get())
    pred_primitive_dontcare = tf.reduce_sum(pred_primitive_annotations, axis=-1) < dont_care_threshold

    # Evaluate vertex metrics
    if args.mode == "triangles" and args.mesh == "scannet" and float(args.simplify) == 1.0:
        gt_mesh = PlyData.read(os.path.join(scene, name + "_vh_clean_2.labels.ply"))

        # Load mapping from vertex to faces
        print("Creating map between faces and vertices...")
        keys = list(gt_mesh["face"].data.dtype.fields.keys())
        if len(keys) != 1:
            print(f"Invalid mesh file: {input_mesh_file}")
            sys.exit(-1)
        face_to_vertices = np.asarray([np.asarray(d) for d in gt_mesh["face"].data[keys[0]]])
        vertex_to_faces_dict = defaultdict(set)
        for face, vertices in enumerate(face_to_vertices):
            for vertex in vertices:
                vertex_to_faces_dict[vertex].add(face)
        max_len = max([len(faces) for faces in vertex_to_faces_dict.values()])
        vertex_to_faces = []
        for vertex in range(len(gt_mesh["vertex"])):
            vertices = list(vertex_to_faces_dict[vertex])
            vertex_to_faces.append(vertices + [-1] * (max_len - len(vertices)))
        vertex_to_faces = np.asarray(vertex_to_faces)
        assert vertex_to_faces.shape[0] == len(gt_mesh["vertex"])

        # Load groundtruth labels
        gt_vertex_annotations = np.asarray(gt_mesh["vertex"].data["label"]).astype("int32")
        gt_vertex_annotations = gt_vertex_annotations - 1
        gt_vertex_annotations = tf.one_hot(gt_vertex_annotations, depth=classes_num)
        gt_face_annotations = tf.gather(gt_vertex_annotations, face_to_vertices)
        gt_face_annotations = tf.reduce_sum(gt_face_annotations, axis=1)
        gt_face_dontcare = tf.reduce_sum(gt_face_annotations, axis=-1) < dont_care_threshold
        gt_face_annotations = tf.one_hot(tf.argmax(gt_face_annotations, axis=-1), depth=classes_num)

        pred_face_annotations = pred_primitive_annotations
        pred_face_dontcare = pred_primitive_dontcare

        # Compute pred annotations
        pred_vertex_annotations = tf.gather(pred_face_annotations, vertex_to_faces) # Out of bound indices are set to 0
        pred_vertex_annotations = tf.reduce_sum(pred_vertex_annotations, axis=1) # Average over all adjacent faces
        pred_vertex_dontcare = tf.reduce_sum(pred_vertex_annotations, axis=-1) < dont_care_threshold
        pred_vertex_annotations = pred_vertex_annotations / tf.reduce_sum(pred_vertex_annotations, axis=-1, keepdims=True)
        pred_vertex_annotations = tf.where(tf.expand_dims(pred_vertex_dontcare, axis=-1), 0.0, pred_vertex_annotations)

        for m in vertex_metrics:
            m.update_state(gt_vertex_annotations, pred_vertex_annotations)

        if not args.debug is None:
            primitive_colors = tf.gather(class_to_color, tf.argmax(pred_face_annotations, axis=-1))
            primitive_colors = tf.cast(primitive_colors, dtype="uint8")
            primitive_colors = tf.where(tf.expand_dims(pred_face_dontcare, axis=-1), 0, primitive_colors)
            mesh.save(os.path.join(args.debug, f"mesh_pred.ply"), primitive_colors)

            primitive_colors = tf.gather(class_to_color, tf.argmax(gt_face_annotations, axis=-1))
            primitive_colors = tf.cast(primitive_colors, dtype="uint8")
            primitive_colors = tf.where(tf.expand_dims(gt_face_dontcare, axis=-1), 0, primitive_colors)
            mesh.save(os.path.join(args.debug, f"mesh_gt.ply"), primitive_colors)

    # Compute metrics for 2d images of fused annotations
    for index, frame in tqdm(list(enumerate(sens.frames))[::args.frames_step]):
        # Render mesh
        camera_to_world = np.linalg.inv(frame.camera_to_world)
        rotation = camera_to_world[:3, :3]
        translation = camera_to_world[:3, 3]
        camera = semantic_meshes.data.Camera(rotation, translation, np.asarray([resolution[1], resolution[0]]), focal_lengths, principal_point)
        primitive_indices, _ = renderer.render(camera)
        primitive_indices = tf.cast(tf.transpose(tf.experimental.dlpack.from_dlpack(primitive_indices), (1, 0)), "int32")

        # Update 2d metrics
        gt_probs = imageio.imread(os.path.join(scene, "label-filt", f"{index}.png"))
        gt_probs = tf.gather(scannet_to_nyu40, tf.cast(gt_probs, "int32"))
        gt_probs = tf.one_hot(gt_probs, depth=classes_num)
        pred_probs = tf.gather(pred_primitive_annotations, primitive_indices, axis=0)
        for m in image_metrics_fused:
            m.update_state(gt_probs, pred_probs)

        if not args.debug is None and index % 100 == 0:
            color = frame.decompress_color_jpeg()
            imageio.imwrite(os.path.join(args.debug, f"{index}_fused.png"), tfcv.util.colorize(segmentation=pred_probs, image=color, class_to_color=class_to_color))

    print(columnar(
        [[scene_index + 1] + [m.result().numpy() for m in vertex_metrics] + [m.result().numpy() for m in image_metrics_network] + [m.result().numpy() for m in image_metrics_fused]],
        ["Scenes"] + [("Vertex-" + m.name) for m in vertex_metrics] + [("ImageNetwork-" + m.name) for m in image_metrics_network] + [("ImageFused-" + m.name) for m in image_metrics_fused],
        no_borders=True
    ))

    if not args.debug is None:
        print("Stopping after first scene due to debug mode")
        break

for prefix, metrics in [("vertex", vertex_metrics), ("image_network", image_metrics_network), ("image_fused", image_metrics_fused)]:
    result["metrics"][prefix] = {}
    for m in metrics:
        value = m.result().numpy()
        if len(value.shape) > 1:
            result["metrics"][prefix][m.name] = value.tolist()
        else:
            result["metrics"][prefix][m.name] = float(value)

if not args.output is None and len(args.output) > 0:
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    i = 1
    while True:
        file = os.path.join(args.output, f"run-{i}.yaml")
        if not os.path.isfile(file):
            break
        i += 1
    print(f"Saving results to {file}")
    with open(file, "w") as f:
        yaml.dump(result, f, default_flow_style=False)

print("Results:")
print(result)
