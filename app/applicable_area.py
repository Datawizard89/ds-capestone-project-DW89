import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gcs_integration as gcs
import matplotlib.colors as mcolors
from dotenv import load_dotenv
import os

load_dotenv()
JSON_KEYFILE_PATH = os.getenv('JSON_KEYFILE_PATH')
BUCKET_NAME = os.getenv('BUCKET_NAME')
CUTOUT_BLOB_BASE = os.getenv('CUTOUT_BLOB_BASE')

# plt.figure(figsize=(8, 8))

# print(CUTOUT_BLOB_BASE)

# BLOB_NAME = CUTOUT_BLOB_BASE + '001_Steglitzer_Sate.png_CO.png'

# ## load images from the cloud
# img = gcs.read_image_from_gcs(BUCKET_NAME , BLOB_NAME, JSON_KEYFILE_PATH)
# print(img)


# load all cutouts:
cutout_names = [
    '001_Steglitzer_Sate.png_CO.png',
    '002_Attilastraße_Sate.png_CO.png',
    '004_Attilastraße_Sate.png_CO.png',
    '005_Attilastraße_Sate.png_CO.png',
    '006_Attilastraße_Sate.png_CO.png',
    '007_Steglitzer_Sate.png_CO.png',
    '008_Attilastraße_Sate.png_CO.png',
    '009_Alt-Lankwitz_Sate.png_CO.png',
    '009_Ritterstraße_Sate.png_CO.png',
    '010_Attilastraße_Sate.png_CO.png',
    '011_Alt-Lankwitz_Sate.png_CO.png',
    '012_Steglitzer_Sate.png_CO.png',
    '013_Steglitzer_Sate.png_CO.png',
    '014_Attilastraße_Sate.png_CO.png',
    '015_Ritterstraße_Sate.png_CO.png',
    '016_Ritterstraße_Sate.png_CO.png',
    '017_Alt-Lankwitz_Sate.png_CO.png',
    '017_Steglitzer_Sate.png_CO.png',
    '020_Steglitzer_Sate.png_CO.png',
    '022_Ritterstraße_Sate.png_CO.png',
    '024_Ritterstraße_Sate.png_CO.png',
    '024_Steglitzer_Sate.png_CO.png',
    '025_Steglitzer_Sate.png_CO.png',
    '026_Ritterstraße_Sate.png_CO.png'
]


COLORS = ['red', 'green', 'blue', 'orange']


def kmeans_photo_generation(cutout_img, num_clusters, colors, index):
    # Get the shape of the image and reshape it for clustering
    height, width, channels = cutout_img.shape
    pixels = cutout_img.reshape(-1, channels)    

    # kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Reshape the labels to match the original image shape
    clustered_image = labels.reshape(height, width) 
    cmap = mcolors.ListedColormap(colors)

    bounds = np.arange(np.min(clustered_image), np.max(clustered_image) + 1, 1)  # Define class boundaries
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=False)
    

    # Create a visualization of the clustered image
    # plt.figure(figsize=(8, 8))
    plt.imshow(clustered_image,  cmap = cmap, norm=norm)


    cbar = plt.colorbar()

    plt.title('K-means Clustering Result')
    plt.axis('off')
    plt.show()
    plt.savefig(f'../data/clustered/{index}.png')

    
    
    return clustered_image

    

def generate_filtered_image(clustered_image):
    class_no = int(input("Enter the class number: "))     

    #clustered image filtered
    clustered_image_filtered = (clustered_image == class_no).astype(int)

    # print(clustered_image_filtered)

     # Reshape the labels to match the original image shape
    cmap = mcolors.ListedColormap(colors=COLORS)

    bounds = np.arange(np.min(clustered_image_filtered), np.max(clustered_image_filtered) + 1, 1)  # Define class boundaries
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=False)
    


    # Create a visualization of the clustered image filtered
    # plt.figure(figsize=(8, 8))
    plt.imshow(clustered_image_filtered,  cmap = cmap, norm=norm)

    cbar = plt.colorbar()

    plt.title('K-means Clustering Result')
    plt.axis('off')
    plt.show()
    plt.savefig(f'../data/clustered/{index}_filtered.png')    

    return clustered_image_filtered






def surface_calculations(clustered_image):
    """
    gets the clustered image and the class name picked by the user and returns the calculations
    """

    height, width = clustered_image.shape
    total_pixels = height*width 
    picked_class_pixels = np.count_nonzero(clustered_image == 1)
    class_zero = np.count_nonzero(clustered_image==0)

    percentage_applicable = round(picked_class_pixels/(total_pixels - class_zero) * 100, 2)

    estmiation_of_size = round(picked_class_pixels * 0.0089,2)

    number_of_panels = round(estmiation_of_size/(1.65*0.99) * 0.60, 0)

    low_band_energy = 250 * number_of_panels

    high_band_energy = 350 * number_of_panels

    print(f'The applicable area is {percentage_applicable}% of the total area which is approximately {estmiation_of_size} sqm. Based on our\
          understanding we might be able to install at lest {number_of_panels} pieces of solar panels sized 1.65m x 0.99m which can produce\
            {low_band_energy} - {high_band_energy} watts of electricty per ????')

 
    





# creating filtered images
for index, val in enumerate(cutout_names[:3]):

    # read images from GCS
    BLOB_NAME = CUTOUT_BLOB_BASE + val
    cutout_img = gcs.read_image_from_gcs(BUCKET_NAME , BLOB_NAME, JSON_KEYFILE_PATH) 

    # kmeans
    plt.figure(figsize=(8, 8))
    clustered_image = kmeans_photo_generation(cutout_img, num_clusters=5, colors=COLORS, index=index)
    # print(clustered_img)

    # generate filtered image
    plt.figure(figsize=(8, 8))
    generate_filtered_image(clustered_image)    

    # created calculations
    surface_calculations(clustered_image)




