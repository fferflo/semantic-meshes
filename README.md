# Semantic Meshes

_A framework for annotating 3D meshes using the predictions of a 2D semantic segmentation model._

---------

## Workflow

1. Reconstruct a mesh of your scene from a set of images using [Colmap](https://github.com/colmap/colmap).
2. Send all undistorted images through your segmentation model (e.g. from [tf-semseg](https://github.com/fferflo/tf-semseg) or [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras)) to produce 2D semantic annotation images.
3. Project all 2D annotations into the 3D mesh and fuse conflicting predictions.
4. Render the annotated mesh from original camera poses to produce new 2D consistent annotation images, or save it as a colorized ply file.

Example output for a traffic scene with annotations produced by a model that was trained on [Cityscapes](https://www.cityscapes-dataset.com/):

![view1](https://github.com/fferflo/semantic-meshes/blob/master/images/view1.jpg)
![view2](https://github.com/fferflo/semantic-meshes/blob/master/images/view2.jpg)

## Usage

We provide a python interface that enables easy integration with numpy and machine learning frameworks like Tensorflow. A full example script is provided in [`colorize_cityscapes_mesh.py`](https://github.com/fferflo/semantic-meshes/blob/master/python/scripts/colorize_cityscapes_mesh.py) that annotates a mesh using a segmentation model that was pretrained on Cityscapes. The model is downloaded automatically and the prediction peformed on-the-fly.

```python
import semantic_meshes

...

# Load a mesh from colmap workspace path and ply mesh file
mesh = semantic_meshes.colmap.ColmapTriangleMesh(args.colmap, args.input_ply)
# Instantiate an aggregator for aggregating the 2D input annotations per 3D primitive
aggregator = semantic_meshes.fusion.MeshAggregator(primitives=mesh.getPrimitivesNum(), classes=19)

...

# Process all input images
for image_file in image_files:
    image = imageio.imread(image_file)
    ...
    # Predict class probability distributions for all pixels in the input image
    prediction = predictor(image)

    ...

    # Render the mesh from the pose of the given image (must be part of the colmap workspace)
    # This returns an image that contains the index of the projected mesh primitive per pixel
    primitive_indices, _ = mesh.render(image_file)

    # Aggregate the class probability distributions of all pixels per primitive
    aggregator.add(primitive_indices, prediction)

# After all images have been processed, the mesh contains a consistent semantic representation of the environment
aggregator.get() # Returns an array that contains the class probability distribution for each primitive

...

# Save colorized mesh to ply
mesh.save(args.output_ply, primitive_colors)
```

## Docker

If you want to skip installation and jump right in, we provide a docker file that can be used without any further steps. Otherwise, see [Installation](#Installation).

1. Install [docker](https://docs.docker.com/engine/install/) and [gpu support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Build the docker image: `docker build -t semantic-meshes https://github.com/fferflo/semantic-meshes.git#master`
   * If your system is using a proxy, add: `--build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...`
3. Open a command prompt in the docker image and mount a folder from your host system (`HOST_PATH`) that contains your colmap workspace into the docker image (`DOCKER_PATH`): `docker run -v /HOST_PATH:/DOCKER_PATH --gpus all -it semantic-meshes bash`
4. Run the provided example script inside the docker image to annotate the mesh with Cityscapes annotations:
```colorize_cityscapes_mesh.py --colmap /DOCKER_PATH/colmap --input_ply /DOCKER_PATH/colmap/dense/meshed-delaunay.ply --images /DOCKER_PATH/colmap/dense/images --output_ply /DOCKER_PATH/colorized_mesh.ply```

Running the repository inside a docker image is significantly slower than running it in the host system (12sec/image vs 2sec/image on RTX 6000).

## Installation

#### Dependencies

* **CUDA**: https://developer.nvidia.com/cuda-downloads
* **OpenMP**: On Ubuntu: `sudo apt install libomp-dev`
* **Python 3**
* **Boost**: Requires the python and numpy components of the Boost library, which have to be compiled for the python version that you are using. If you're lucky, your OS ships compatible Boost and Python3 versions. Otherwise, [compile boost from source](https://www.boost.org/doc/libs/1_76_0/more/getting_started/unix-variants.html) and make sure to include the `--with-python=python3` switch.

#### Build

The repository contains CMake code that builds the project and provides a python package in the build folder that can be installed using [pip](https://pypi.org/project/pip/).

CMake downloads, builds and installs all other dependencies automatically. If you don't want to clutter your global system directories, add `-DCMAKE_INSTALL_PREFIX=...` to install to a local directory.

The framework has to be compiled for specific number of classes (e.g. 19 for Cityscapes, or 2 for a binary segmentation). Add a semicolon-separated list with `-DCLASSES_NUMS=2;19;...` for all number of classes that you want to use. A longer list will significantly increase the compilation time.

An example build:

```
git clone https://github.com/fferflo/semantic-meshes
cd semantic-meshes
mkdir build
mkdir install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DCLASSES_NUMS=19 ..
make -j8
make install # Installs to the local install directory
pip install ./python
```
