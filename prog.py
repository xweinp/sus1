import os, sys
import numpy as np
from PIL import Image, ImageOps


from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def clusters_to_files(y, names):
    clusters = {}
    for i, name in enumerate(names):
        yi = int(y[i])
        if yi not in clusters:
            clusters[yi] = []
        clusters[yi].append(name)

    # text file:

    with open('clusters.txt', 'w') as f:
        for cluster in clusters:
            for name in clusters[cluster]:
                f.write(f"{name} ")
            f.write("\n")
    
    # html file:

    with open('clusters.html', 'w') as f:
        for cluster in clusters:
            for name in clusters[cluster]:
                f.write(f"<img src='training_samples/{name}'>")
            f.write("<HR>")

    print("Clusters saved to clusters.txt and clusters.html")

def center_image_by_centroid(img):
    img = img.astype(float)
    h, w = img.shape

    # Zamiana obrazu na binarny: 1 dla pikseli > 0, 0 dla pozostałych
    binary_img = (img > 0).astype(float)

    y, x = np.indices((h, w))

    total = binary_img.sum()
    if total == 0:
        return img.copy()

    # Obliczenie centroidu na podstawie binarnego obrazu
    cx = (x * binary_img).sum() / total
    cy = (y * binary_img).sum() / total

    target_cx = w // 2
    target_cy = h // 2

    shift_x = int(round(target_cx - cx))
    shift_y = int(round(target_cy - cy))

    shifted = np.roll(img, shift=shift_y, axis=0)
    shifted = np.roll(shifted, shift=shift_x, axis=1)

    if shift_y > 0:
        shifted[:shift_y, :] = 0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0

    if shift_x > 0:
        shifted[:, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0

    return shifted


def sizes_np(image):
    return np.array(image.shape, dtype=np.float32)

def metric_sz(X, squared=False):
    X = np.asarray(X, dtype=np.float32)
    
    l = X[:, None, :]
    lj = X[None, :, :]
    
    r = np.maximum(l, lj) / np.minimum(l, lj)
    R = np.prod(r, axis=2)
    
    return R

def transform_image(image, max_x, max_y):
    pad_x = max_x - image.shape[0]
    pad_y = max_y - image.shape[1]
    # pad with white pixels (0)
    # add information about the number of pixels
    extra_features = np.array([
        image.shape[0],
        image.shape[1],
        # np.sum(np.any(image > 10, axis=0))
        # image.shape[0] * image.shape[1]
        # 
    ], dtype=np.float32)

    image = np.pad(
        image,
        ((0, pad_x), (0, pad_y)),
        mode='constant',
        constant_values=0
    )

    # center image by centroid
    image = center_image_by_centroid(image)

    image = image.flatten()
    return image.astype(np.float32), extra_features



def make_size_clusters(images):
    max_x = max([image.shape[0] for image in images])
    max_y = max([image.shape[1] for image in images])

    X_sizes = [sizes_np(image) for image in images]
    X_sizes = np.array(X_sizes)

    sils = []
    ags = []

    for n_clusters in range(2, 10):
        ag = AgglomerativeClustering(
            n_clusters=n_clusters,
            method='ward',
            metric='euclidean'
        )
        y = ag.fit_predict(X_sizes)
        sil = silhouette_score(X_sizes, y)
        
        sils.append(sil)
        ags.append(ag)

    best_clusters = np.argmax(sils)
    ag = ags[best_clusters]
    size_clusters = ag.fit_predict(X_sizes)

    return size_clusters

def make_clusters(images, image_names):
    max_x = max([image.shape[0] for image in images])
    max_y = max([image.shape[1] for image in images])

    def make_row(image):
        image = np.pad(
            image,
            ((0, max_x - image.shape[0]), (0, max_y - image.shape[1])),
            mode='constant',
            constant_values=0
        )
        image = center_image_by_centroid(image)
        return image.flatten()
    
    X = [
        make_row(image)
        for image in images
    ]

    X = np.array(X, dtype=np.float32)

    n_samples = X.shape[0]
    n_components = min(max_x * max_y, 20, n_samples - 1)

    if n_components <= 1:
        # not enough samples
        return [{
            'cluster': 0,
            'name': name
        } for name in image_names]


    tr = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=10
    )

    X = tr.fit_transform(X)

    min_cl, max_cl = 2, min(n_samples, 80)
    sils = []
    ags = []

    # TODO: wypróbuj kilk arózńych algorytmów klasteryzacji (w koncu po cos jest te 10 minut...)

    for i in range(min_cl, max_cl):
        ag = KMeans(n_clusters=i, algorithm='elkan')
        y = ag.fit_predict(X)

        sils.append(silhouette_score(X, y))
        ags.append(ag)

    if max(sils) < 0:
        # all in one cluster
        return [{
            'cluster': 0,
            'name': name
        } for name in image_names]
    
    
    best_clusters = np.argmax(sils)
    ag = ags[best_clusters]
    y = ag.fit_predict(X)

    return y




def main():
    file = sys.argv[1]

    with open(file, 'r') as f:
        image_names = f.readlines()
    
    images = []
    extra_features = []

    for filename in image_names:
        image = Image.open(filename)
        image = ImageOps.grayscale(image)
        image = image.invert() # I want 0 to be white
        image = np.array(image)
        images.append(image)

    # First I cluster the images by size.
    # Then I will perform clustering in each size cluster.
    size_clusters = make_size_clusters(images)
    
    n_clusters = 0
    save_clusters = []
    save_names = []

    for cluster in size_clusters:
        cluster_images = [images[i] for i in range(len(images)) if size_clusters[i] == cluster]
        cluster_names = [image_names[i] for i in range(len(images)) if size_clusters[i] == cluster]

        # Now I will perform clustering in each size cluster.
        new_clusters = make_clusters(cluster_images, cluster_names)
        save_clusters += [i + n_clusters for i in new_clusters]
        n_clusters += len(set(new_clusters))
        save_names += cluster_names

    clusters_to_files(save_clusters, save_names)

if __name__ == "__main__":
    main()
    