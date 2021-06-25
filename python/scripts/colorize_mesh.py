#!/usr/bin/env python3

import argparse, os, imageio, sys, semantic_meshes
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from distinctipy import distinctipy

parser = argparse.ArgumentParser(description="Annotate a colmap mesh with classes from mask images and save as colorized ply.")
parser.add_argument("--colmap", type=str, required=True, help="Path to colmap workspace folder containing {cameras, images, points3D}.{bin|txt}")
parser.add_argument("--input_ply", type=str, required=True, help="Input mesh file")
parser.add_argument("--masks", type=str, required=True, help="Path to folder containing masks of images reconstructed in the colmap workspace")
parser.add_argument("--classes", type=int, required=True, help="Number of classes")
parser.add_argument("--output_ply", type=str, required=True, help="Output mesh file")
parser.add_argument("--remap", action="store_true", help="Indicate that the input masks are color images, remap the set of unique colors to a contiguous range of class indices")
args = parser.parse_args()

if args.remap:
    # Define a color-to-class dictionary for remapping
    color_to_class = {}
    def get_class_for_color(color):
        color = tuple(color.tolist())
        if color in color_to_class:
            return color_to_class[color]
        else:
            next_class = len(color_to_class)
            color_to_class[color] = next_class
            return next_class

print("Creating mesh...")
mesh = semantic_meshes.data.Ply(args.input_ply)
colmap_workspace = semantic_meshes.data.Colmap(args.colmap)
renderer = semantic_meshes.render.triangles(mesh)
aggregator = semantic_meshes.fusion.MeshAggregator(primitives=renderer.getPrimitivesNum(), classes=args.classes)

print("Annotating mesh...")
mask_files = os.listdir(args.masks)
mask_files = [os.path.join(args.masks, file) for file in mask_files if file.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))]
for mask_file in tqdm(mask_files):
    # Load input image
    mask = imageio.imread(mask_file)

    # Remap colors to class indices
    if args.remap:
        assert len(mask.shape) == 2 or len(mask.shape) == 3
        color_channels = mask.shape[2] if len(mask.shape) == 3 else 1
        assert color_channels == 1 or color_channels == 3
        shape = mask.shape[:2]

        mask = np.reshape(mask, (-1, color_channels))

        unique_colors, inv_mapping = np.unique(mask, axis=0, return_inverse=True)
        unique_classes = np.array([get_class_for_color(c) for c in unique_colors])

        mask = unique_classes[inv_mapping]
        mask = np.reshape(mask, shape)

    assert len(mask.shape) == 2
    assert np.all(0 <= mask) and np.all(mask < args.classes)

    # Aggregator expects probabilities, so we convert class indices in the mask image to one-hot vectors
    probs = tf.one_hot(mask, depth=args.classes)

    # Project annotations into mesh
    primitive_indices, _ = renderer.render(colmap_workspace.getCamera(mask_file))
    probs = tf.transpose(probs, perm=(1, 0, 2))
    aggregator.add(primitive_indices, probs)

if args.remap:
    # Use color mapping that was extracted from the original image
    class_to_color = np.array([[0, 0, 0]] * args.classes)
    for color, class_index in color_to_class.items():
        class_to_color[class_index] = color
    class_to_color = class_to_color.astype("uint8")
    print(f"Found {len(color_to_class)} unique colors: {[class_to_color[c].tolist() for c in sorted(list(color_to_class.values()))]}")
else:
    # Generate a distinct color mapping for the given classes
    class_to_color = np.asarray(distinctipy.get_colors(args.classes)) * 255.0
    print(f"Generated {len(color_to_class)} unique colors: {[class_to_color[c].tolist() for c in range(args.classes)]}")

print("Computing primitive colors...")
primitive_annotations = aggregator.get()
dont_care_threshold = 0.9
without_annotations = tf.reduce_sum(primitive_annotations, axis=-1) < dont_care_threshold # Primitives that did not receive annotations from any projection

primitive_annotations = tf.argmax(primitive_annotations, axis=-1) # Choose class with highest aggregated probability
primitive_colors = tf.gather(class_to_color, primitive_annotations) # Convert class indices to colors
primitive_colors = tf.cast(primitive_colors, dtype="uint8")
primitive_colors = tf.where(tf.expand_dims(without_annotations, axis=-1), 0, primitive_colors) # Set color for pixels without annotations to black

print("Saving colorized mesh...")
mesh.save(args.output_ply, primitive_colors)
