#!/usr/bin/env python3

import argparse, os, yaml, math, tfcv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from copy import deepcopy
from collections import defaultdict

parser = argparse.ArgumentParser(description="Create graphs for finished scannet evaluations")
parser.add_argument("--path", type=str, required=True, help="Path to where scannet evaluation .yaml files are stored")
args = parser.parse_args()

plotargs = {"marker": "o", "markersize": 6}
fontsize = 15

print("Reading yaml files...")

params = []
for yaml_file in [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith(".yaml")]:
    with open(yaml_file, "r") as f:
        p = yaml.safe_load(f)
    assert "path" not in p["params"]
    p["path"] = yaml_file
    if "aggregator" not in p["params"]:
        p["params"]["aggregator"] = "sum"
    if "simplify" not in p["params"]:
        p["params"]["simplify"] = 1.0
    else:
        p["params"]["simplify"] = float(p["params"]["simplify"])
    if "mesh" not in p["params"]:
        p["params"]["mesh"] = "scannet"
    if "num" not in p["params"]:
        p["params"]["num"] = 100
    if "offset" not in p["params"]:
        p["params"]["offset"] = 0
    if "texel_resolution" not in p["params"]:
        assert p["params"]["mode"] == "triangles"
        p["params"]["texel_resolution"] = 0.0
    for m in ["image_fused", "image_network", "vertex"]:
        for k in list(p["metrics"][m].keys()):
            if k.startswith("confusion_matrix"):
                p["metrics"][m]["ConfusionMatrix"] = np.asarray(p["metrics"][m][k])
                del p["metrics"][m][k]
    params.append(p)

def equals(p1, p2, root=True):
    if root:
        p1 = deepcopy(p1["params"])
        p2 = deepcopy(p2["params"])
        del p1["num"]
        del p1["offset"]
        del p2["num"]
        del p2["offset"]
    return p1 == p2

i1 = 0
while i1 < len(params):
    i2 = i1 + 1
    while i2 < len(params):
        if equals(params[i1], params[i2]):
            print("Added " + str(params[i2]) + " to " + str(params[i1]))
            if params[i1]["params"]["offset"] == params[i2]["params"]["offset"]:
                print("Same offset at:")
                print(params[i1]["path"])
                print(params[i1]["params"])
                print(params[i1]["metrics"]["image_fused"]["Accuracy"])
                print(params[i2]["path"])
                print(params[i2]["params"])
                print(params[i2]["metrics"]["image_fused"]["Accuracy"])
                sys.exit(-1)
            if "Accuracy" in params[i1]["metrics"]["image_fused"]:
                del params[i1]["metrics"]["image_fused"]["Accuracy"]
                del params[i1]["metrics"]["image_fused"]["MeanIoU"]
                del params[i1]["metrics"]["image_network"]["Accuracy"]
                del params[i1]["metrics"]["image_network"]["MeanIoU"]
                del params[i1]["metrics"]["vertex"]["Accuracy"]
                del params[i1]["metrics"]["vertex"]["MeanIoU"]
            params[i1]["metrics"]["image_fused"]["ConfusionMatrix"] += params[i2]["metrics"]["image_fused"]["ConfusionMatrix"]
            params[i1]["metrics"]["image_network"]["ConfusionMatrix"] += params[i2]["metrics"]["image_network"]["ConfusionMatrix"]
            # params[i1]["metrics"]["vertex"]["ConfusionMatrix"] += params[i2]["metrics"]["vertex"]["ConfusionMatrix"]
            params[i1]["params"]["num"] += params[i2]["params"]["num"]
            del params[i2]
        else:
            i2 += 1
    i1 += 1

drop_indices = []
for i, p in enumerate(params):
    if p["params"]["mesh"] == "scannet" and p["params"]["num"] < 100:
        drop_indices.append(i)
        print("Dropping " + str(p["params"]))
for i in reversed(drop_indices):
    del params[i]

for p in params:
    if "Accuracy" not in p["metrics"]["image_fused"]:
        p["metrics"]["image_fused"]["Accuracy"] = tfcv.metric.confusionmatrix_to_accuracy(p["metrics"]["image_fused"]["ConfusionMatrix"])
        p["metrics"]["image_network"]["Accuracy"] = tfcv.metric.confusionmatrix_to_accuracy(p["metrics"]["image_network"]["ConfusionMatrix"])
        # p["metrics"]["vertex"]["Accuracy"] = tfcv.metric.confusionmatrix_to_accuracy(p["metrics"]["vertex"]["ConfusionMatrix"])


used_params = set()
def print_plot_params(name, params):
    print(f"Plot: {name}")
    for p in params:
        acc = p["metrics"]["image_fused"]["Accuracy"]
        print("    " + os.path.basename(p["path"]) + " " + str(p["params"]) + f" {acc * 100.0:.3f}")
        used_params.add(p["path"])





print("Creating plots...")
def all_plots():
    plt.gcf().subplots_adjust(left=0.2, right=0.9, bottom=0.15, top=0.95)

images_equal_weights = [1.0] # 0.0, 0.5,
plot_params = [p for p in params if p["params"]["frames_step"] == 1 and p["params"]["simplify"] == 1.0 and p["params"]["mesh"] == "scannet" and p["params"]["images_equal_weight"] in images_equal_weights and p["params"]["aggregator"] == "mul"]
plot_params = sorted(plot_params, key=lambda p: p["params"]["texel_resolution"])
print_plot_params("texelres_to_accuracy", plot_params)
plt.figure(1)
plt.rcParams.update({'font.size': fontsize})
for images_equal_weight in set([p["params"]["images_equal_weight"] for p in plot_params]):
    if 0.0 <= images_equal_weight and images_equal_weight <= 1.0:
        ps = [p for p in plot_params if p["params"]["images_equal_weight"] == images_equal_weight]
        xs = [(p["params"]["texel_resolution"]) for p in ps]
        ys = [p["metrics"]["image_fused"]["Accuracy"] for p in ps]
        if images_equal_weight == 0:
            label = f"$w_P$"
        elif images_equal_weight == 1.0:
            label = f"$w_I$"
        else:
            label = f"${1.0 - images_equal_weight} w_P + {images_equal_weight} w_I$"
        plt.plot(xs, ys, **plotargs) # , label=label
#plt.plot(xs, [p["metrics"]["image_network"]["Accuracy"] for p in plot_params], label="Network prediction")
plt.xlabel("Texel resolution $\gamma$")
plt.ylabel("Pixel accuracy")
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x * 100.0:.2f}%"))
ticks = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
plt.xticks(ticks, [str(t) for t in ticks])
#plt.title("Pixel accuracy")
plt.gca().set_xlim(left=0)
# plt.legend()
all_plots()
plt.savefig(os.path.join(args.path, "texelres_to_accuracy.png"), dpi=300) # , bbox_inches="tight"
plt.close(1)

plot_params = [p for p in params if p["params"]["frames_step"] == 1 and p["params"]["mode"] == "triangles" and p["params"]["images_equal_weight"] <= 1.0 and p["params"]["aggregator"] == "mul" and p["params"]["mesh"] == "scannet" and p["params"]["simplify"] == 1.0]
plot_params = sorted(plot_params, key=lambda p: p["params"]["images_equal_weight"])
print_plot_params("imagesequalweight_to_accuracy", plot_params)
xs = [p["params"]["images_equal_weight"] for p in plot_params]
plt.figure(1)
plt.rcParams.update({'font.size': fontsize})
plt.plot(xs, [p["metrics"]["image_fused"]["Accuracy"] for p in plot_params], **plotargs) # , label="Fused annotation"
plt.xticks([0.0, 1.0], labels=[r"$w^{(P)}$", r"$w^{(I)}$"])
plt.ylabel("Pixel accuracy")
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x * 100.0:.1f}%"))
#plt.title("Pixel accuracy")
#plt.legend()
all_plots()
plt.savefig(os.path.join(args.path, "imagesequalweight_to_accuracy.png"), dpi=300)
plt.close(1)

plot_params = [p for p in params if p["params"]["frames_step"] == 1 and p["params"]["mode"] == "triangles" and (p["params"]["images_equal_weight"] == 0.0 or p["params"]["images_equal_weight"] == 1.0) and p["params"]["simplify"] == 1.0 and p["params"]["mesh"] == "scannet"]
print_plot_params("aggregators", plot_params)
aggregators = list(set(p["params"]["aggregator"] for p in plot_params))
ys_wi = {p["params"]["aggregator"]: p["metrics"]["image_fused"]["Accuracy"] for p in plot_params if p["params"]["images_equal_weight"] == 1.0} # [plot_params[0]["metrics"]["image_network"]["Accuracy"]] +
ys_wi = np.asarray([(ys_wi[agg] if agg in ys_wi else float("nan")) for agg in aggregators])
ys_wp = {p["params"]["aggregator"]: p["metrics"]["image_fused"]["Accuracy"] for p in plot_params if p["params"]["images_equal_weight"] == 0.0}
ys_wp = np.asarray([(ys_wp[agg] if agg in ys_wp else float("nan")) for agg in aggregators])
# xs = np.arange(len(aggregators))
# low = min(ys_wi.tolist() + ys_wp.tolist())
# high = max(ys_wi.tolist() + ys_wp.tolist())
# plt.figure(1)
# plt.rcParams.update({'font.size': fontsize})
# plt.ylim([low - 0.5 * (high - low), high + 0.5 * (high - low)])
# bar_width = 0.1
# plt.bar(xs + bar_width * 0, ys_wp, bar_width, label="$w = w^{(P)}$")
# plt.bar(xs + bar_width * 1, ys_wi, bar_width, label="$w = w^{(I)}$")
# plt.xticks(np.array(range(len(xs))) + bar_width * len(aggregators) / 4, aggregators)
# plt.legend()
# plt.ylabel("Pixel accuracy")
# plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x * 100.0:.1f}%"))
# all_plots()
# plt.savefig(os.path.join(args.path, "aggregators.png"), dpi=300, bbox_inches="tight")
# plt.close(1)
with open(os.path.join(args.path, "aggregators.csv"), "w") as f:
    f.write("aggregator,accuracy_w_i,accuracy_w_p\n")
    for agg, y_wi, y_wp in zip(aggregators, ys_wi, ys_wp):
        f.write(f"{agg},{y_wi},{y_wp}\n")

plot_params = []
plot_params_by_texelres_by_simplify = defaultdict(lambda: defaultdict(list))
simplify_keys = set()
texel_resolutions = [0.0, 0.2] # [0.0, 0.1, 0.2, 0.3, 0.5]
for p in params:
    if p["params"]["frames_step"] == 1 and p["params"]["images_equal_weight"] == 1.0 and p["params"]["aggregator"] == "mul" and p["params"]["mesh"] == "scannet" and p["params"]["texel_resolution"] in texel_resolutions:
        plot_params.append(p)
        plot_params_by_texelres_by_simplify[p["params"]["texel_resolution"]][p["params"]["simplify"]] = p
        simplify_keys.add(p["params"]["simplify"])
simplify_keys = sorted(list(simplify_keys))
print_plot_params("simplify_to_accuracy", plot_params)
xs = np.asarray(simplify_keys)
ys_all = {}
for texel_resolution, d1 in sorted(list(plot_params_by_texelres_by_simplify.items()), key=lambda t: t[0]):
    ys_all[texel_resolution] = np.asarray([(d1[s]["metrics"]["image_fused"]["Accuracy"] if s in d1 else float("nan")) for s in simplify_keys])
plt.figure(1)
plt.rcParams.update({'font.size': fontsize})
for texel_resolution, ys in ys_all.items():
    plt.plot(np.log(xs), ys, label=f"$\gamma={texel_resolution}$", **plotargs)
# plt.plot(xs, [(plot_params2[x]["metrics"]["image_fused"]["Accuracy"] if x in plot_params2 else float("nan")) for x in xs], label="texelres=0.0", **plotargs)
plt.xlabel("Proportion of triangles")
plt.ylabel("Pixel accuracy")
baseline = plot_params[0]["metrics"]["image_network"]["Accuracy"]
# plt.axhline(y=baseline, color=np.array([0.1, 1.0, 0.1]), linestyle='-', lw=1, label="Network")
plt.plot(np.log(np.asarray([xs[0], xs[-1]])), [baseline, baseline], label="Baseline", linestyle="--", color=np.array([1.0, 0.1, 0.1]))
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{np.exp(x):.2f}"))
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: f"{y * 100.0:.1f}%"))
ticks = np.power(10, np.array([0, 1, 2]))
plt.xticks(np.log(1.0 / ticks), [f"{1.0 / t * 100.0:.1f}%" for t in ticks]) # (f"$\\frac{{1}}{{ {t} }}$" if t > 1 else "1")
plt.gca().set_xlim(right=0)
#plt.title("Pixel accuracy")
plt.legend()
all_plots()
plt.savefig(os.path.join(args.path, "simplify_to_accuracy.png"), dpi=300)
plt.close(1)

plot_params = [p for p in params if p["params"]["mode"] == "triangles" and p["params"]["images_equal_weight"] == 1.0 and p["params"]["aggregator"] == "mul" and p["params"]["mesh"] == "scannet" and p["params"]["simplify"] == 1.0]
plot_params = sorted(plot_params, key=lambda p: p["params"]["frames_step"])
print_plot_params("framesstep_to_accuracy", plot_params)
xs = [1.0 / p["params"]["frames_step"] for p in plot_params]
plt.figure(1)
plt.rcParams.update({'font.size': fontsize})
plt.plot(np.log(xs), [p["metrics"]["image_fused"]["Accuracy"] for p in plot_params], **plotargs) # , label="Fused annotation"
plt.xlabel("Proportion of frames")
plt.ylabel("Pixel accuracy")
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x * 100.0:.1f}%"))
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{np.exp(x) * 100.0:.1f}%"))
ticks = np.power(10, np.array([0, 1, 2]))
plt.xticks(np.log(1.0 / ticks), [f"{1.0 / t * 100.0:.1f}%" for t in ticks]) # (f"$\\frac{{1}}{{ {t} }}$" if t > 1 else "1")
plt.gca().set_xlim(right=0)
#plt.title("Pixel accuracy")
#plt.legend()
all_plots()
plt.savefig(os.path.join(args.path, "framesstep_to_accuracy.png"), dpi=300)
plt.close(1)

unused_params = [p for p in params if not p["path"] in used_params]
if len(unused_params) > 0:
    print("Unused params:")
    for p in unused_params:
        print("    " + os.path.basename(p["path"]) + " " + str(p["params"]))
