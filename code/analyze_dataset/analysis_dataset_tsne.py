import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from sklearn.manifold import TSNE

dataset = load_from_disk("datasets/http-header-split-embedded-data-v1")

print(dataset)

testset = dataset["test"].select(range(10000))

major_class_label_feature = dataset["test"].features["major_class"]
print(major_class_label_feature)

print("Filtering datasets by class...")
c1 = testset.filter(lambda x: x["major_class"] == 0, num_proc=12)["array"]
c2 = testset.filter(lambda x: x["major_class"] == 1, num_proc=12)["array"]
c3 = testset.filter(lambda x: x["major_class"] == 2, num_proc=12)["array"]
c4 = testset.filter(lambda x: x["major_class"] == 3, num_proc=12)["array"]
c5 = testset.filter(lambda x: x["major_class"] == 4, num_proc=12)["array"]

concat = np.concatenate((c1, c2, c3, c4, c5), axis=0)

print("Fitting TSNE...")
tsne = TSNE(n_components=3, learning_rate="auto", init="pca")
t = tsne.fit_transform(concat)

print(t.shape)

c1s = t[: len(c1)]
c2s = t[len(c1) : len(c1) + len(c2)]
c3s = t[len(c1) + len(c2) : len(c1) + len(c2) + len(c3)]
c4s = t[len(c1) + len(c2) + len(c3) : len(c1) + len(c2) + len(c3) + len(c4)]
c5s = t[len(c1) + len(c2) + len(c3) + len(c4) :]

print(len(c1))
print(len(c2))
print(len(c3))
print(len(c4))
print(len(c5))
print(len(c1s))
print(len(c2s))
print(len(c3s))
print(len(c4s))
print(len(c5s))


print("Plotting figure...")
fig = plt.figure(figsize=(20, 20), dpi=200)
ax = fig.add_subplot(projection="3d")
ax.set_xlabel("t-SNE-1")
ax.set_ylabel("t-SNE-2")
ax.set_zlabel("t-SNE-3")

colormap = ["r", "g", "b", "c", "m"]

for index, transformed_class in enumerate([c1s, c2s, c3s, c4s, c5s]):
    ax.scatter(
        alpha=0.25,
        xs=transformed_class[:, 0],
        ys=transformed_class[:, 1],
        zs=transformed_class[:, 2],
        c=colormap[index],
        label=major_class_label_feature.int2str(index),
    )

plt.legend()

plt.savefig("plots/tsne_3d_scatterplot", bbox_inches="tight", pad_inches=0)
