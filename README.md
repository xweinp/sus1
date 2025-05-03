# Overview

The way that I found to work best for this project is to perform the clustering twice:

1. I cluster the images based only on sizees of the images
2. I perform a separate clustering inside each of the size clusters

This had to be done because I needed to extract features using nonlinear dimentionality reduction. The features that I want to extract are very different for big sins containg several characters and small cahracters like, eg. a dot.

For the size clustering I went with agglomerative clustering using ward's method. 

I reduce the dimensionality inside of each cluster with LLE (locally linear embedding). The maximum nober of components is 20 though it may be smaller for very small clusters.

For the second clustering I used KMeans with elkan algorithm. 

For all lustering I check a range of number of clusters I want to divide the data into. I use the silhouette score to determine the best number of clusters.

I tested several approaches (without the size clustering, with PCA, UMAP, Isomap, ... as reduction methods and other clustering methods with different metrics). Some methods worked better with eg. small characters and were able to separate dots form commas very well, others worked better for many characters on one image. 

This approch however works very well on single characters, quite well for 2 - character images, ok for dots, commas etc but is bad for big images with many characters. Unfortunatelly I didn't have time to make it work better :(

So finally I ended up with:
- using <<AgglomerativeClustering >> with best number of clusters from range [2, 10) selected using silhouette score
- On each cluster:
    - I apply transformatinos to the images (put them on grayscle, ivert so that 0 is white and 255 is black, pad to the same size and move them  so that the center of image (calculated with assuming pixels can be only white if pixel = 0 or black if pixel > 0) is in the center of space
    - reduce the dimensionality with <<LocallyLinearEmbedding>> with n_components = min(max_x * max_y, 20, n_samples - 1), where max_x and max_y are the maximum width and height of the images in the cluster. I set n_neighbors = min(10, n_components)
    - use <<KMeans>> with elkan algorith and best number of clusters from range [2, min(n_samples, 80)) selected using silhouette score

# Performance

On my PC (Ryzen 5 8400f, 16GB RAM) the programme runs pretty fast. 

For the 7600 images provided it task description: 
- size clustering takes < 10 seconds
- final clustering takes up to 30 seconds per size cluster, but it strongly depends on the number ofimages in that cluster.
- the whole process takes less than a minute

# Usage

To run the code you need to first make the virtual environment. To do taht you have to run:
```bash
./make_env.sh
```
On some systems you may need to run:
```bash
chmod +x make_env.sh
```

Then you can run the code with:
```bash
python3 prog.py <path_to_file>
```
where <path_to_file> is the path to the file containing paths to images.

You can also enable the logging by setting <log> to <True> at the top of <prog.py>.

The results are saved in the <results> directory. The programme creates the directory by itself. The directory is created if it does not exist. If the directory already contains files they will be overwritten. Files other than <clusters.txt> and <clusters.html> won't be affected.

