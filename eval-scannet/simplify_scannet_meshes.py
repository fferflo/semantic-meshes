#!/usr/bin/env python3

import argparse, os, tempfile, subprocess, tqdm
from plyfile import PlyData

parser = argparse.ArgumentParser(description="Simplify meshes in the scannet dataset by the given factor")
parser.add_argument("--scannet", type=str, required=True, help="Path to scannet directory")
parser.add_argument("--factor", action="append", type=float, help="Fraction of number of faces in the new mash")
args = parser.parse_args()


filter_script_in = """
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Simplification: Quadric Edge Collapse Decimation">
  <Param value="FACES_NUM" name="TargetFaceNum" tooltip="The desired final number of faces." isxmlparam="0" type="RichInt" description="Target number of faces"/>
  <Param value="0" name="TargetPerc" tooltip="If non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial size." isxmlparam="0" type="RichFloat" description="Percentage reduction (0..1)"/>
  <Param value="0.3" name="QualityThr" tooltip="Quality threshold for penalizing bad shaped faces.&lt;br>The value is in the range [0..1]&#xa; 0 accept any kind of face (no penalties),&#xa; 0.5  penalize faces with quality &lt; 0.5, proportionally to their shape&#xa;" isxmlparam="0" type="RichFloat" description="Quality threshold"/>
  <Param value="false" name="PreserveBoundary" tooltip="The simplification process tries to do not affect mesh boundaries during simplification" isxmlparam="0" type="RichBool" description="Preserve Boundary of the mesh"/>
  <Param value="1" name="BoundaryWeight" tooltip="The importance of the boundary during simplification. Default (1.0) means that the boundary has the same importance of the rest. Values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border. Admitted range of values (0,+inf). " isxmlparam="0" type="RichFloat" description="Boundary Preserving Weight"/>
  <Param value="false" name="PreserveNormal" tooltip="Try to avoid face flipping effects and try to preserve the original orientation of the surface" isxmlparam="0" type="RichBool" description="Preserve Normal"/>
  <Param value="false" name="PreserveTopology" tooltip="Avoid all the collapses that should cause a topology change in the mesh (like closing holes, squeezing handles, etc). If checked the genus of the mesh should stay unchanged." isxmlparam="0" type="RichBool" description="Preserve Topology"/>
  <Param value="true" name="OptimalPlacement" tooltip="Each collapsed vertex is placed in the position minimizing the quadric error.&#xa; It can fail (creating bad spikes) in case of very flat areas. &#xa;If disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices. " isxmlparam="0" type="RichBool" description="Optimal position of simplified vertices"/>
  <Param value="false" name="PlanarQuadric" tooltip="Add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh, as a side effect, more triangles will be preserved in flat areas (allowing better shaped triangles)." isxmlparam="0" type="RichBool" description="Planar Simplification"/>
  <Param value="0.001" name="PlanarWeight" tooltip="How much we should try to preserve the triangles in the planar regions. If you lower this value planar areas will be simplified more." isxmlparam="0" type="RichFloat" description="Planar Simp. Weight"/>
  <Param value="false" name="QualityWeight" tooltip="Use the Per-Vertex quality as a weighting factor for the simplification. The weight is used as a error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified." isxmlparam="0" type="RichBool" description="Weighted Simplification"/>
  <Param value="true" name="AutoClean" tooltip="After the simplification an additional set of steps is performed to clean the mesh (unreferenced vertices, bad faces, etc)" isxmlparam="0" type="RichBool" description="Post-simplification cleaning"/>
  <Param value="false" name="Selected" tooltip="The simplification is applied only to the selected set of faces.&#xa; Take care of the target number of faces!" isxmlparam="0" type="RichBool" description="Simplify only selected faces"/>
 </filter>
</FilterScript>
"""

class RunException(BaseException): # TODO: this somewhere else?
    def __init__(self, message, code):
        self.message = message
        self.code = code

def run(command, log=None, echo=True):
    if not isinstance(command, list):
        command = command.split(" ")
    if echo:
        print("> " + " ".join(command))
    if not log is None:
        log.write("> " + " ".join(command) + "\n")
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
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
        msg = result + "\nFailed to run " + " ".join(command) + ". Got return code " + str(process.returncode)
        raise RunException(msg, process.returncode)
    return result



# Search for scenes
scenes = sorted([os.path.join(args.scannet, "scans", f) for f in os.listdir(os.path.join(args.scannet, "scans")) if f.startswith("scene")])
print(f"Found {len(scenes)} scenes in {args.scannet}")

for factor in args.factor:
    print(f"Simplifying meshes with factor {factor}")
    for scene in tqdm.tqdm(scenes):
        mesh = PlyData.read(os.path.join(scene, os.path.basename(scene) + "_vh_clean_2.labels.ply"))
        faces_num = int(factor * mesh["face"].data.shape[0])
        input_mesh = os.path.join(scene, os.path.basename(scene) + "_vh_clean_2.labels.ply")
        output_mesh = os.path.join(scene, os.path.basename(scene) + f"_vh_clean_2.labels_simplified-{factor}.ply")

        with tempfile.NamedTemporaryFile() as tmp:
            filter_script = filter_script_in.replace("FACES_NUM", f"{faces_num}")
            tmp.write(filter_script.encode())
            tmp.flush()

            run(["bash", "-c", f"xvfb-run -a -s \"-screen 0 800x600x24\" meshlabserver -i {input_mesh} -s {tmp.name} -o {output_mesh}"], echo=False)
