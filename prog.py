import os, sys
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.base import clone


log = False

tqdm = tqdm if log else lambda x: x

def clusters_to_files(clusters):
    
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # text file:

    with open('./results/clusters.txt', 'w') as f:
        for cluster in clusters:
            for name in cluster:
                f.write(os.path.basename(name) + " ")
            f.write("\n")
    
    # html file:

    with open('./results/clusters.html', 'w') as f:
        for cluster in clusters:
            for name in cluster:
                f.write(f"<img src='{name}'>\n")
            f.write("<HR>\n")

    if log: print("Clusters saved to clusters.txt and clusters.html")



def center_image_by_centroid(img):
    img = img.astype(float)
    h, w = img.shape

    binary_img = (img > 0).astype(float)

    y, x = np.indices((h, w))

    total = binary_img.sum()
    if total == 0:
        return img.copy()

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



def make_size_clusters(images):
    if log: print("Clustering by size...")
    X_sizes = [sizes_np(image) for image in images]
    X_sizes = np.array(X_sizes)

    sils = []
    ags = []

    for n_clusters in tqdm(range(2, 10)):
        ag = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        y = ag.fit_predict(X_sizes)
        sil = silhouette_score(X_sizes, y)
        
        sils.append(sil)
        ags.append(ag)

    best_clusters = np.argmax(sils)
    ag = ags[best_clusters]
    y = ag.fit_predict(X_sizes)
    size_clusters = []
    for i in range(best_clusters + 2):
        size_clusters.append(np.where(y == i)[0])

    if log: print("Number of size clusters:", len(size_clusters))

    return size_clusters

def make_clusters(images, indexes):
    if log: print("Clustering by content...")
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
        make_row(images[i])
        for i in indexes
    ]

    X = np.array(X, dtype=np.float32)

    n_samples = X.shape[0]
    n_components = min(max_x * max_y, 20, n_samples - 1)

    if n_components <= 1:
        # not enough samples
        if log: print("Putting all in one cluster")
        return [[0 for i in range(n_samples)]] 


    tr = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=min(n_components, 10)
    )

    X = tr.fit_transform(X)

    min_cl, max_cl = 2, min(n_samples, 80)
    sils = []
    models = []

    def add_model(model):
        y = model.fit_predict(X)
        sil = silhouette_score(X, y)
        sils.append(sil)
        models.append(model)

    for n_clusters in range(min_cl, max_cl):
        add_model(KMeans(
            n_clusters=n_clusters, 
            algorithm='elkan',
            max_iter=1000
        ))
        
        
    if max(sils) < 0:
        # all in one cluster
        if log: print("Putting all in one cluster")
        return [[0 for i in range(n_samples)]] 
    
    best_clusters = np.argmax(sils)
    model =clone(models[best_clusters])
    y = model.fit_predict(X)

    n_labels = len(np.unique(y))
    clusters = [np.where(y == i)[0] for i in range(n_labels)]
    clusters = [[indexes[i] for i in cluster] for cluster in clusters]

    if log: print("Number of clusters:", n_labels)
    return clusters



def main():
    file = sys.argv[1]

    if log: print("Reading image paths from:", file)

    with open(file, 'r') as f:
        image_names = f.readlines()
        image_names = [name.strip() for name in image_names]
    
    images = []

    for filename in image_names:
        image = Image.open(filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image) # I want 0 to be white
        image = np.array(image) / 255.0
        images.append(image)

    # First I cluster the images by size.
    # Then I will perform clustering in each size cluster.
    size_clusters = make_size_clusters(images)

    result_clusters = []

    for cluster in tqdm(size_clusters):
        # Now I will perform clustering in each size cluster.
        new_clusters = make_clusters(images, cluster)
        result_clusters.extend(new_clusters)

    result_clusters = [
        [image_names[i] for i in cluster]
        for cluster in result_clusters
    ]
    if log: print("Final number of clusters:", len(result_clusters))
    clusters_to_files(result_clusters)

if __name__ == "__main__":
    main()
    