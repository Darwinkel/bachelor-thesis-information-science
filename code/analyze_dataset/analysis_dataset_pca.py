import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from scipy import spatial
from sklearn.decomposition import PCA

dataset = load_from_disk("datasets/http-header-split-embedded-data-v1")

print(dataset)

sample1 = dataset["test"][1]
sample2 = dataset["test"][35]

cosine_similarity = spatial.distance.cosine(sample1["array"], sample2["array"])


class_label_feature = dataset["train"].features["minor_class"]
major_class_label_feature = dataset["train"].features["major_class"]

print(
    f"Minor labels: {class_label_feature.int2str(sample1['minor_class'])} | "
    f"{class_label_feature.int2str(sample2['minor_class'])}"
)
print(
    f"Major labels: {major_class_label_feature.int2str(sample1['major_class'])} | "
    f"{major_class_label_feature.int2str(sample2['major_class'])}"
)
print(f"Similarity: {cosine_similarity}")

print(major_class_label_feature)


testset = dataset["test"].select(range(100000))

print("Filtering datasets by class...")
c1 = testset.filter(lambda x: x["major_class"] == 0, num_proc=12)["array"]
c2 = testset.filter(lambda x: x["major_class"] == 1, num_proc=12)["array"]
c3 = testset.filter(lambda x: x["major_class"] == 2, num_proc=12)["array"]
c4 = testset.filter(lambda x: x["major_class"] == 3, num_proc=12)["array"]
c5 = testset.filter(lambda x: x["major_class"] == 4, num_proc=12)["array"]

print("Fitting PCA...")
pca = PCA(n_components=3)
pca.fit(np.concatenate((c1, c2, c3, c4, c5), axis=0))

print("Transforming PCA...")
c1s = pca.transform(c1)
c2s = pca.transform(c2)
c3s = pca.transform(c3)
c4s = pca.transform(c4)
c5s = pca.transform(c5)


print("Plotting figure...")
fig = plt.figure(figsize=(20, 20), dpi=200)
ax = fig.add_subplot(projection="3d")
ax.set_xlabel("PCA-1")
ax.set_ylabel("PCA-2")
ax.set_zlabel("PCA-3")

ax.scatter(
    alpha=0.25,
    xs=c1s[:, 0],
    ys=c1s[:, 1],
    zs=c1s[:, 2],
    c="r",
    label=major_class_label_feature.int2str(0),
)
ax.scatter(
    alpha=0.25,
    xs=c2s[:, 0],
    ys=c2s[:, 1],
    zs=c2s[:, 2],
    c="g",
    label=major_class_label_feature.int2str(1),
)
ax.scatter(
    alpha=0.25,
    xs=c3s[:, 0],
    ys=c3s[:, 1],
    zs=c3s[:, 2],
    c="b",
    label=major_class_label_feature.int2str(2),
)
ax.scatter(
    alpha=0.25,
    xs=c4s[:, 0],
    ys=c4s[:, 1],
    zs=c4s[:, 2],
    c="c",
    label=major_class_label_feature.int2str(3),
)
ax.scatter(
    alpha=0.25,
    xs=c5s[:, 0],
    ys=c5s[:, 1],
    zs=c5s[:, 2],
    c="m",
    label=major_class_label_feature.int2str(4),
)

plt.legend()

plt.savefig("plots/pca_3d_scatterplot", bbox_inches="tight", pad_inches=0)
