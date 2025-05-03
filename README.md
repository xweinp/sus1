## Overview

The approach that worked best for this project involves performing clustering in two stages:

1. **Initial Clustering by Image Size**  
   First, I cluster the images based solely on their dimensions (width and height).

2. **Secondary Clustering within Each Size Cluster**  
   Within each size-based cluster, I perform a second clustering based on image content.

This two-step process is necessary because I extract features using nonlinear dimensionality reduction. The features I want to capture differ significantly between large images (which may contain multiple characters) and small ones (such as a single dot).

For the initial size-based clustering, I use **Agglomerative Clustering** with **Ward's method**.

Inside each size cluster:
- I apply **Locally Linear Embedding (LLE)** for dimensionality reduction. The number of components is capped at 20, but may be lower for small clusters.
- Then, I use **KMeans** with the **Elkan algorithm** for clustering.

For all clustering steps, I test a range of possible cluster counts and select the best using the **silhouette score**.

I tested multiple alternatives (e.g., skipping size clustering, using PCA, UMAP, Isomap, and various clustering algorithms with different metrics). Some methods performed better on small characters—separating dots from commas well—while others worked better on images with multiple characters.

This final approach performs very well on single characters, reasonably well on two-character images, and adequately on dots and commas. However, it performs poorly on large images with many characters. Unfortunately, I didn’t have time to improve that part.

### Final Pipeline

- Use `AgglomerativeClustering` with the best number of clusters selected from the range [2, 10), based on the silhouette score.
- For each size-based cluster:
  - Convert images to grayscale.
  - Invert so that white = 0 and black = 255.
  - Pad to a uniform size and center the content (assuming pixels are white if pixel = 0, black otherwise).
  - Apply `LocallyLinearEmbedding` with  
    `n_components = min(max_x * max_y, 20, n_samples - 1)`  
    where `max_x` and `max_y` are the maximum width and height of images in the cluster.  
    Set `n_neighbors = min(10, n_components)`.
  - Run `KMeans` with the Elkan algorithm, selecting the best number of clusters from the range `[2, min(n_samples, 80))`, using the silhouette score.

## Performance

On my PC (Ryzen 5 8400f, 16GB RAM), the program runs efficiently.

For the 7,600 images provided in the task description:
- Size clustering takes less than 10 seconds.
- Final clustering takes up to 30 seconds per size cluster, depending on the number of images.
- The whole process completes in under one minute.

## Usage


To run the code, first create a virtual environment.

On some systems, you may need to make the script executable first:

```bash
chmod +x make_env.sh
```

To create the virtual environment and install all necessary libraries, run:

```bash
./make_env.sh
```


Then run the program with:

```bash
python3 prog.py <path_to_file>
```

Where `<path_to_file>` is the path to a file containing the image paths.

To enable logging, set `log = True` at the top of `prog.py`.

The results are saved in the `results` directory. The program will create this directory if it does not exist. If it already exists, only `clusters.txt` and `clusters.html` will be overwritten. Other files will not be affected.


## Example

```bash
chmod +x make_env.sh
./make_env.sh
python3 prog.py ./example_set/list.txt
```

Now you can find clusters in the `results` directory!

