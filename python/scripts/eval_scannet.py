#!/usr/bin/env python3

import argparse, os, imageio, sys, tf_semseg, semantic_meshes, yaml
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from collections import defaultdict
from distinctipy import distinctipy
from columnar import columnar

parser = argparse.ArgumentParser(description="Evaluate semantic-meshes on the scannet dataset")
parser.add_argument("--scannet", type=str, required=True, help="Path to scannet directory")
parser.add_argument("--images_equal_weight", type=float, default=0.5, help="Soft boolean flag such that 0.0 weights images equally and 1.0 weights pixels equally in the fusion step")
parser.add_argument("--frames_step", type=int, default=1, help="Only use every n-th frame from a scene's frames")
parser.add_argument("--debug", type=str, default=None, help="A path where debugging images and plys are stored for the first scene")
parser.add_argument("--output", type=str, default=None, help="File where results will be stored")

parser.add_argument("--mode", type=str, default="triangles", help="Argument determining whether 'triangles' or 'texels' meshes should be used for the fusion")
parser.add_argument("--texel_resolution", type=float, default=0.1, help="Texel resolution parameter. Only valid with '--mode texels'")

args = parser.parse_args()
assert args.mode in ["triangles", "texels"]

dont_care_threshold = 0.9
classes_num = 40
class_to_color = np.asarray(distinctipy.get_colors(classes_num)) * 255.0

# Load pretrained model
model = tf_semseg.model.pretrained.tuinicr.esanet_resnet_v1b_34_nbt1d_nyuv2.create()
predictor = lambda x: model(x, training=False)
preprocess = tf_semseg.model.pretrained.tuinicr.esanet_resnet_v1b_34_nbt1d_nyuv2.preprocess

# Search for scenes
scenes = sorted([os.path.join(args.scannet, "scans", f) for f in os.listdir(os.path.join(args.scannet, "scans")) if f.startswith("scene")])
print(f"Found {len(scenes)} scenes in {args.scannet}")

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
    tf_semseg.metric.Accuracy(classes_num=classes_num, dontcare_prediction="error"),
    tf_semseg.metric.MeanIoU(classes_num=classes_num, dontcare_prediction="error")
]
image_metrics_network = [
    tf_semseg.metric.Accuracy(classes_num=classes_num, dontcare_prediction="forbidden"),
    tf_semseg.metric.MeanIoU(classes_num=classes_num, dontcare_prediction="forbidden")
]
image_metrics_fused = [
    tf_semseg.metric.Accuracy(classes_num=classes_num, dontcare_prediction="error"),
    tf_semseg.metric.MeanIoU(classes_num=classes_num, dontcare_prediction="error")
]
for scene_index, scene in enumerate(scenes):
    # Load scannet scene
    name = os.path.basename(scene)
    sens = semantic_meshes.data2.SensFile(os.path.join(scene, name + ".sens"), max_frames=None)
    resolution = np.array([sens.color_height, sens.color_width])
    input_mesh_file = os.path.join(scene, name + "_vh_clean_2.labels.ply")
    gt_mesh = PlyData.read(input_mesh_file)

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
    aggregator = semantic_meshes.fusion.MeshAggregator(primitives=renderer.getPrimitivesNum(), classes=classes_num, aggregator="sum", images_equal_weight=args.images_equal_weight)

    # Load mapping from vertex to faces
    print("Creating map between faces and vertices...")
    face_to_vertices = np.asarray([np.asarray(d) for d in gt_mesh["face"].data["vertex_indices"]])
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

    print(f"Loaded scene with {renderer.getPrimitivesNum()} mesh-primitives and {len(sens.frames)} frames")

    # Fuse predictions on mesh
    for index, frame in tqdm(list(enumerate(sens.frames))[::args.frames_step]):
        # Render mesh
        camera_to_world = np.linalg.inv(frame.camera_to_world)
        rotation = camera_to_world[:3, :3]
        translation = camera_to_world[:3, 3]
        camera = semantic_meshes.data.Camera(rotation, translation, np.asarray([resolution[1], resolution[0]]), focal_lengths, principal_point)
        primitive_indices, depth = renderer.render(camera)
        depth = tf.expand_dims(tf.transpose(tf.experimental.dlpack.from_dlpack(depth), (1, 0)), axis=-1) * 1000.0
        primitive_indices = tf.transpose(tf.experimental.dlpack.from_dlpack(primitive_indices), (1, 0))

        # Predict classes
        color = frame.decompress_color_jpeg()
        color_preprocessed = tf.image.resize(color, (480, 640), method="bilinear")
        depth_preprocessed = tf.image.resize(depth, (480, 640), method="nearest")
        color_preprocessed, depth_preprocessed = preprocess(color_preprocessed, depth_preprocessed)
        pred_probs = predictor([np.expand_dims(color_preprocessed, axis=0), np.expand_dims(np.expand_dims(depth_preprocessed, axis=0), axis=-1)])[0]
        pred_probs = tf.image.resize(pred_probs, resolution, method="bilinear")

        # Update 2d metrics
        gt_probs = imageio.imread(os.path.join(scene, "label-filt", f"{index}.png"))
        gt_probs = tf.gather(scannet_to_nyu40, tf.cast(gt_probs, "int32"))
        gt_probs = tf.one_hot(gt_probs, depth=classes_num)
        for m in image_metrics_network:
            m.update_state(gt_probs, pred_probs)

        # Aggregate
        primitive_indices = tf.experimental.dlpack.to_dlpack(tf.transpose(primitive_indices, (1, 0))) # TODO: try remove dlpack
        aggregator.add(primitive_indices, tf.transpose(pred_probs, (1, 0, 2)))

        if not args.debug is None:
            imageio.imwrite(os.path.join(args.debug, f"{index}_color.png"), color)
            imageio.imwrite(os.path.join(args.debug, f"{index}_depth.png"), tf.where(tf.math.is_finite(depth), depth, 0).numpy())
            imageio.imwrite(os.path.join(args.debug, f"{index}_gt.png"), tf_semseg.util.colorize(segmentation=gt_probs, image=color, class_to_color=class_to_color))
            imageio.imwrite(os.path.join(args.debug, f"{index}_pred.png"), tf_semseg.util.colorize(segmentation=pred_probs, image=color, class_to_color=class_to_color))

    print("Computing primitive annotations...")
    pred_primitive_annotations = aggregator.get()
    pred_primitive_dontcare = tf.reduce_sum(pred_primitive_annotations, axis=-1) < dont_care_threshold

    # Evaluate vertex metrics
    if args.mode == "triangles":
        pred_face_annotations = pred_primitive_annotations
        pred_face_dontcare = pred_primitive_dontcare

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

        if not args.debug is None:
            color = frame.decompress_color_jpeg()
            imageio.imwrite(os.path.join(args.debug, f"{index}_fused.png"), tf_semseg.util.colorize(segmentation=pred_probs, image=color, class_to_color=class_to_color))

    print(columnar(
        [[scene_index + 1] + [m.result().numpy() for m in vertex_metrics] + [m.result().numpy() for m in image_metrics_network] + [m.result().numpy() for m in image_metrics_fused]],
        ["Scenes"] + [("Vertex-" + m.name) for m in vertex_metrics] + [("ImageNetwork-" + m.name) for m in image_metrics_network] + [("ImageFused-" + m.name) for m in image_metrics_fused],
        no_borders=True
    ))

    if not args.debug is None:
        print("Stopping after first scene due to debug mode")
        break


result = {"metrics": {}, "params": {}}
for prefix, metrics in [("vertex", vertex_metrics), ("image_network", image_metrics_network), ("image_fused", image_metrics_fused)]:
    result["metrics"][prefix] = {}
    for m in metrics:
        result["metrics"][prefix][m.name] = float(m.result().numpy())
result["params"]["images_equal_weight"] = args.images_equal_weight
result["params"]["frames_step"] = args.frames_step
result["params"]["mode"] = args.mode
if args.mode == "texels":
    result["params"]["texel_resolution"] = args.texel_resolution

if not args.output is None and len(args.output) > 0:
    print(f"Saving results to {args.output}")
    with open(args.output, "w") as f:
        yaml.dump(result, f, default_flow_style=False)

print("Results:")
print(result)
