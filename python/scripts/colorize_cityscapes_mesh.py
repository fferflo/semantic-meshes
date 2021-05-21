#!/usr/bin/env python3

import argparse, os, imageio, sys
import semantic_meshes.fusion, semantic_meshes.colmap
import tf_semseg
import tensorflow as tf
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Annotate a colmap mesh with cityscapes classes and save as colorized ply.")
parser.add_argument("--colmap", type=str, required=True, help="Path to colmap workspace folder containing {cameras, images, points3D}.bin")
parser.add_argument("--input_ply", type=str, required=True, help="Input mesh file")
parser.add_argument("--images", type=str, required=True, help="Path to folder containing all images reconstructed in the colmap workspace")
parser.add_argument("--output_ply", type=str, required=True, help="Output mesh file")
args = parser.parse_args()

# Maps cityscapes class indices to RGB colors
class_to_color = [
    (128, 64,128),
    (244, 35,232),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32)
]

print("Loading pretrained segmentation model...")
predictor = tf_semseg.model.pretrained.vladkryvoruchko.pspnet_resnet_v1s_101_cityscapes() # Load a pretrained model
predictor = tf_semseg.predict.sliding(predictor, (713, 713), 0.2) # Perform sliding window inference
predictor = tf_semseg.predict.multi_scale(predictor, [0.5]) # Resize input images to resolution that matches Cityscapes dataset
predictor = tf.function(predictor) # Improve performance by compiling to tensorflow graph

print("Creating mesh...")
mesh = semantic_meshes.colmap.ColmapTriangleMesh(args.colmap, args.input_ply)
aggregator = semantic_meshes.fusion.MeshAggregator(primitives=mesh.getPrimitivesNum(), classes=19)

print("Annotating mesh...")
image_files = os.listdir(args.images)
image_files = [os.path.join(args.images, file) for file in image_files]
for image_file in tqdm(image_files):
    # Load input image
    image = imageio.imread(image_file) # Loads in RGB order

    # Predict segmentation
    image = image - np.array([123.68, 116.779, 103.939]) # Preprocess: Subtract imagenet mean
    image = image[:, :, ::-1] # Preprocess: RGB -> BGR, since pretrained network requires BGR input
    image = np.expand_dims(image, axis=0) # Add batch dimension
    prediction = predictor(image)
    prediction = prediction[0, :, :, :] # Remove batch dimension

    # Project annotations into mesh
    primitive_indices, _ = mesh.render(image_file)
    prediction = tf.transpose(prediction, perm=(1, 0, 2))
    aggregator.add(primitive_indices, prediction)

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
