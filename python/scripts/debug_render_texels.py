#!/usr/bin/env python3

import argparse, os, semantic_meshes, math, imageio, tfcv
import tensorflow as tf
import numpy as np
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser(description="Render texels on a single triangle and save to files")
parser.add_argument("--output", type=str, required=True, help="Output folder")
args = parser.parse_args()

if not os.path.isdir(args.output):
    os.makedirs(args.output)

# Initialize tensorflow context
tf.zeros(1)

# Create mesh
vertex = np.array([
    (0.4, 0, 0),
    (0.5, 1, 0),
    (0.6, 0, 0)
], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

for face in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
    name = "-".join([str(s) for s in face])
    face = np.array([
        (face,),
    ], dtype=[("vertex_indices", "i4", (3,))])

    ply = PlyData([
        PlyElement.describe(vertex, "vertex"),
        PlyElement.describe(face, "face")
    ], text=True)

    mesh_file = os.path.join(args.output, f"mesh-{name}.ply")
    if os.path.isfile(mesh_file):
        os.remove(mesh_file)
    with open(mesh_file, mode="wb") as f:
        ply.write(f)

    # Load mesh
    mesh = semantic_meshes.data.Ply(mesh_file)

    # Create camera
    eye = np.asarray([-0.5, -0.5, 4.0])
    target = np.asarray([-0.5, -0.5, 0.0])
    up = np.asarray([0.0, 1.0, 0.0])
    import pyrr
    camera_to_world = np.transpose(pyrr.matrix44.create_look_at(eye, target, up, dtype="float32"), (1, 0))
    camera_to_world = np.linalg.inv(camera_to_world)
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]

    field_of_view_y = math.radians(45.0)
    resolution = np.asarray([4000, 4000])
    principal_point = resolution.astype("float32") / 2.0
    focal_lengths = np.asarray([
        principal_point[0] / (resolution[0] / resolution[1] * math.tan(field_of_view_y / 2.0)),
        principal_point[1] / math.tan(field_of_view_y / 2.0)
    ])

    camera = semantic_meshes.data.Camera(rotation, translation, np.asarray([resolution[1], resolution[0]]), focal_lengths, principal_point)

    # Render
    renderer = semantic_meshes.render.texels(mesh, [camera], 0.01)
    primitive_indices, depth = renderer.render(camera)
    depth = tf.expand_dims(tf.transpose(tf.experimental.dlpack.from_dlpack(depth), (1, 0)), axis=-1)
    primitive_indices = tf.cast(tf.transpose(tf.experimental.dlpack.from_dlpack(primitive_indices), (1, 0)), "int32")

    classes_num = (tf.reduce_max(primitive_indices) + 1).numpy()
    sidelength = int(-0.5 + math.sqrt(0.25 + 2 * classes_num))
    print(f"Has {classes_num} texels and sidelength {sidelength}")
    primitive_indices = tf.where(primitive_indices >= 0, primitive_indices, classes_num)

    # Save
    depth = tf.where(tf.math.is_inf(depth), 0.0, depth)
    imageio.imwrite(os.path.join(args.output, f"depth-{name}.png"), depth.numpy())

    class_to_color = []
    i = 0
    n = 1
    flip = True
    for c in range(classes_num):
        if flip:
            class_to_color.append([255, 201, 14])
        else:
            class_to_color.append([0, 162, 232])
        i += 1
        if i == n:
            i = 0
            n += 1
            flip = n % 2 == 1
        else:
            flip = not flip

    color = tfcv.util.colorize(primitive_indices, class_to_color=class_to_color, classes_num=classes_num, dont_care_color=[255, 255, 255])
    imageio.imwrite(os.path.join(args.output, f"color-{name}.png"), color.numpy().astype("uint8"))
