import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gcs_integration as gcs
import matplotlib.colors as mcolors
from dotenv import load_dotenv
import os
import streamlit as st
import cv2

load_dotenv()
JSON_KEYFILE_PATH = os.getenv('JSON_KEYFILE_PATH')
BUCKET_NAME = os.getenv('BUCKET_NAME')
CUTOUT_BLOB_BASE = os.getenv('CUTOUT_BLOB_BASE')



####################################################load data from address mapping ##############################
# import json
# with open('../data/addresses.json', 'r') as f:
#     address_mapping = json.load(f)

# full_addresses = []
# for index, item in enumerate(address_mapping):
#     full_addresses.append(item['address'])


# print(full_addresses)



# STREAMLIT
st.title('Solarmente Berlin App')



# Get the original photos from the trained folder
satelite_images = '..\\data\\pesentation\\satelite_images'
image_files = [f for f in os.listdir(satelite_images) if f.lower().endswith(('.png'))]





# cutout_images = '..\\data\\cutouts'
# cutout_images_files = [f for f in os.listdir(cutout_images) if f.lower().endswith(('.png'))]
# test_img = cutout_images_files[item_index]


# print([item for item in cutout_images if test_img in cutout_images])



# Dropdown 

st.markdown("""
### Instruction:\n
__1.__ Please select the address you want to make calculations on from the below drowpdown.\n
__2.__ Push the segmentation button and let Solarmente does the magic for you and separate the surfaces\n
__3.__ Pick the surface code you need\n

""")

st.markdown("________________________________________________________________")


st.markdown("### Select an Address")
selected_image = st.selectbox(label='', options=image_files)
# print(selected_image)
# st.write(f"You selected: {selected_image}")


# get the index of list of satelite images to apply it on list of cutouts
if selected_image in image_files:
    image_index = image_files.index(selected_image)
    print(f"The index of {selected_image} is {image_index}")
else:
    print(f"{selected_image} not found in the list")





# load the cutouts and display them
cutouts = '..\\data\\pesentation\\cutouts'
cutout_files = [f for f in os.listdir(cutouts) if f.lower().endswith(('.png'))]


# split to two columns
left_column, right_column = st.columns(2)




# divider


left_column.markdown("#### The Property's Sattelite View")

# Display satelite image
if selected_image:
    image_path = os.path.join(satelite_images, selected_image)
    left_column.image(image_path, caption=selected_image, use_column_width=True)
else:
    left_column.write("Please select an image from the dropdown.")


cutout_selected = cutout_files[image_index]


# Display the cutouts
right_column.markdown("#### The Masked View of the Property")
if cutout_selected:
    cutout_path = os.path.join(cutouts, cutout_selected)
    right_column.image(cutout_path, caption=cutout_selected, use_column_width=True)
else:
    right_column.write("Please select an image from the dropdown.")















COLORS = ['red', 'green', 'blue', 'orange']
cutout_np = cv2.imread(cutout_path)


def kmeans_photo_generation(cutout_np, cutout_image, num_clusters, colors):
    # Get the shape of the image and reshape it for clustering
    height, width, channels = cutout_np.shape
    pixels = cutout_np.reshape(-1, channels)    

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
    plt.figure(figsize=(8, 8))
    plt.imshow(clustered_image,  cmap = cmap, norm=norm)


    cbar = plt.colorbar()

    plt.title('K-means Clustering Result')
    plt.axis('off')
    
    clustered_filename = f'..\\data\\pesentation\\clustered\\{cutout_image}_clustered.png' 
    plt.savefig(clustered_filename)

    return clustered_filename


   




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
            {low_band_energy} - {high_band_energy} watts of electricty per hour.')

 
    



# generate the kmeans clustering image
cutout_image = cutout_files[image_index]
print(cutout_image)
# plt.figure(figsize=(8, 8))
if st.checkbox('Do you really want to predict?'):
    clustered_file = kmeans_photo_generation(cutout_np, cutout_image, num_clusters=5, colors=COLORS)
    st.write(clustered_file)
    st.image(clustered_file, use_column_width=True)

### ###################################################find clustered images in a directory        
# def list_files_in_directory(folder_path):
#     file_names = []
    
#     # Check if the folder path exists
#     if os.path.exists(folder_path):
#         # List all files in the folder
#         files = os.listdir(folder_path)
        
#         # Filter out only files (excluding directories)
#         file_names = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
#     else:
#         print("Folder path does not exist.")

#     return file_names





# Specify the folder path

# clustered_path = "../data/pesentation/clustered"

# all_clustered = list_files_in_directory(clustered_path)






# create and display the kmeans clustered image
left_column.markdown("#### K-Means clustering Initial Outcome")


# selected_clustered = all_clustered[image_index]

# print(selected_clustered)   
# clsutered_path = os.path.join(clustered_path, selected_clustered)

# st.image(clustered_file, use_column_width=True)





#########################################################################



















































st.markdown("""
### Pick the Surface You Want to Apply the Panels to:\n
__1.__ Each color is related to a surface. By looking at the colored bar next to the above photo, pick the surface number. 
            

""")

st.markdown("________________________________________________________________")

#### drop down for the class selection

selected_class = st.selectbox(label="Pick a surface:", options=[1,2,3,4,5])



##### Generate filtered clusters

def generate_filtered_image(clustered_image, class_no):


    #clustered image filtered

    clustered_image_filtered = (clustered_image == class_no).astype(int)

    # print(clustered_image_filtered)

     # Reshape the labels to match the original image shape
    cmap = mcolors.ListedColormap(colors=['red'])

    bounds = np.arange(np.min(clustered_image_filtered), np.max(clustered_image_filtered) + 1, 1)  # Define class boundaries
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=False)
    


    # Create a visualization of the clustered image filtered
    plt.figure(figsize=(8, 8))
    plt.imshow(clustered_image_filtered,  cmap = cmap, norm=norm)

    cbar = plt.colorbar()
    
    plt.title('K-means Clustering Result')
    plt.axis('off')
    plt.show()
    filtered_filename = f'../data/clustered/_filtered.png'
    plt.savefig(filtered_filename)    

    return filtered_filename

    # plt.figure(figsize=(8, 8))
    # plt.imshow(clustered_image,  cmap = cmap, norm=norm)


    # cbar = plt.colorbar()

    # plt.title('K-means Clustering Result')
    # plt.axis('off')
    
    # clustered_filename = f'..\\data\\pesentation\\clustered\\{cutout_image}_clustered.png' 
    # plt.savefig(clustered_filename)

    return clustered_image_filtered



# generate_filtered_image(clustered_file, selected_class)# plt.figure(figsize=(8, 8))
clustered_np = cv2.imread(clustered_file)

if st.checkbox('Do you really want to refine?'):
    filtered_file = generate_filtered_image(clustered_np, selected_class)
    # st.write(filtered_file )
    st.image(filtered_file, use_column_width=True)