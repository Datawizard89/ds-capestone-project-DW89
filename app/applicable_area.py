###################################################################################################
############################## Impofrting Libaries ################################################
###################################################################################################
# Image liabries
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import io
import numpy as np 
# ML liabries 
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_sample_image
# from yellowbrick.cluster import KElbowVisualizer
# reading env files for requeirements and stuff 
from dotenv import load_dotenv
# Working with dir and such 
import os
# streamlit for app application #here loca 
import streamlit as st
# importing own function 
import gcs_integration as gcs

############################################ System requirement (temporary) ###################################
is_apple = False

###################################################################################################
###################loading requirments via get env ################################################
###################################################################################################
load_dotenv()                                                                                      # loading .env file
JSON_KEYFILE_PATH               = os.getenv('JSON_KEYFILE_PATH')                                   # getting key file path
BUCKET_NAME                     = os.getenv('BUCKET_NAME')                                         # getting bucket name (google drive)
CUTOUT_BLOB_BASE                = os.getenv('CUTOUT_BLOB_BASE')                                    # getting cutout dir
###################loading data from address mapping ###############################################
# import json
# with open('../data/addresses.json', 'r') as f:
#     address_mapping = json.load(f)
# full_addresses = []
# for index, item in enumerate(address_mapping):
#     full_addresses.append(item['address'])
# print(full_addresses)
###################################################################################################
##################################### STREAMLIT ###################################################
###################################################################################################
st.title('Solarmente Berlin App')                                                                  # setting titel of streamlit
################## LISTING original photos from the trained folder#################################
if is_apple == True:
    satelite_images                 = './data/pesentation/satelite_images'
else:
    satelite_images                 = '.\\data\\pesentation\\satelite_images'     
                               # defining dir for lisr
image_files = sorted([f for f in os.listdir(satelite_images) if f.lower().endswith(('.png'))])             # list for dropdown manue later on 
###################################################################################################
# cutout_images = './data/cutouts'
# cutout_images_files = [f for f in os.listdir(cutout_images) if f.lower().endswith(('.png'))]
# test_img = cutout_images_files[item_index]
# print([item for item in cutout_images if test_img in cutout_images])
###################################### UI Instruction ##############################################
st.markdown("""
### Instruction:\n
__1.__ Please select the address you want to make calculations on from the below drowpdown.\n
__2.__ Push the segmentation button and let Solarmente does the magic for you and separate the surfaces\n
__3.__ Pick the surface code you need\n
""")                                                                                                # Giving user instruction markdown like j-nb
##################################### Visible seperater ############################################
st.markdown("________________________________________________________________")                     # drawing a line
#################################### Dropdown org. Images ##########################################
st.markdown("### Select an Address")
selected_image                  = st.selectbox(label='', options=image_files)                       # acttual drowpdown
# print(selected_image)
# st.write(f"You selected: {selected_image}")
#################################### Getting CutOut ###############################################
#########get the index of list of satelite images to apply it on list of cutouts ##################
if selected_image in image_files:                                                                   # getting the index of selectted image, will be used later on for selecting cutouts     
    image_index = image_files.index(selected_image)
    print(f"The index of {selected_image} is {image_index}")
else:
    print(f"{selected_image} not found in the list")        
############################ load the cutouts and display them######################################
cutouts = './data/pesentation/cutouts'
cutout_files                    = sorted([f for f in os.listdir(cutouts) if f.lower().endswith(('.png'))])  # listing cutouts
############################# split to two columns##################################################
left_column, right_column       = st.columns(2)                                                     # so images can be next to each other
############################## divider #############################################################
left_column.markdown("#### The Property's Sattelite View")
############################## Display satelite image ##############################################
if selected_image:
    image_path                 = os.path.join(satelite_images, selected_image)                      # creating path
    left_column.image(image_path, caption=selected_image, use_column_width=True)                    # displaying image left 
else:
    left_column.write("Please select an image from the dropdown.")
############################## Display masked image ################################################
cutout_selected                = cutout_files[image_index]                                          # choosing image via index
# Display the cutouts
right_column.markdown("#### The Masked View")
if cutout_selected:
    cutout_path                = os.path.join(cutouts, cutout_selected)                             # creating path
    right_column.image(cutout_path, caption=cutout_selected, use_column_width=True)
else:
    right_column.write("Please select an image from the dropdown.")
cutout_np                      = cv2.imread(cutout_path)                                            # reading the iamge 
############################## defining colors for cmap ###############################################
COLORS = ['red', 'green', 'blue', 'orange']

#######################################################################################################################
###################################  DEFINING FUNCTIONS ###############################################################
#######################################################################################################################
###################################  Modelling function ###############################################################
#######################################################################################################################
@st.cache_data(persist="disk")
def kmeans_photo_generation(cutout_np, cutout_image, num_clusters, colors):
    ################################### OLD just KMEAN #############################################
    ################################################################################################
    # Get the shape of the image and reshape it for clustering
    # height, width, channels = cutout_np.shape
    # pixels = cutout_np.reshape(-1, channels)    
    # # kmeans clustering
    # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    # labels = kmeans.labels_
    # cluster_centers = kmeans.cluster_centers_
    ########################## Start Refinement of KMEAN via KNN ###################################
    ################################################################################################
    #image processing
    height, width, channels         = cutout_np.shape                                                       # gettting shape of image 
    pixels                          = cutout_np.reshape(-1, channels)                                       # Image flatting 3D to 2D 
    #Kmean 
    kmeans                          = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)           # K-means clustering and fitting
    kmeans_labels                   = kmeans.labels_                                                        # Getting K-mean labeles
    cluster_centers                 = kmeans.cluster_centers_                                               # getting centers of KMEAN 
    #model = KMeans()
    # visualizer                      = KElbowVisualizer(kmeans, k=(1, 11))                                   # Instantiate the Elobow visualizer
    # KNN 
    n_neighbors                     = 10                                                                    # Number of nearest neighbors to consider for KNN
    knn                             = NearestNeighbors(n_neighbors=n_neighbors, 
                                                       metric='euclidean').fit(pixels)                      # Initiating KNN and fitting pixel for refinement 
    knn_labels                      = knn.kneighbors(pixels, return_distance=False).mean(axis=1)            # getting lables
    # Voting of models
    combined_labels                 = np.column_stack((kmeans_labels, knn_labels))                          # Combine K-means and KNN labels for refined clustering
    combined_labels                 = np.round(combined_labels).astype(int)                                 # Making sure labeles after combi are in format int
    final_labels                    = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                       axis=1, arr=combined_labels)                                         # Calculate the final cluster labels based on a majority vote
    segmented_image                 = final_labels.reshape(cutout_np.shape[:2])                             # Reshape the final labels back into the shape of the original image               
    # st.write(np.max(segmented_image))
    # Visualize the segmented image
    cmap                            = mcolors.ListedColormap(colors)                                        # creating color map for visualization
    bounds                          = np.arange(np.min(segmented_image), np.max(segmented_image) + 1, 1)    # Define class boundaries
    norm                            = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=False)
    # plotting the image
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image, cmap='nipy_spectral')
    # plt.imshow(segmented_image)
    cbar                            = plt.colorbar()
    plt.title('KNN Refinement')
    plt.axis('off')
    plt.show()
    # saving the image 
    if is_apple == True:
        clustered_filename               = f'./data/pesentation/clustered/{cutout_image}_clustered.png' 
    else:
        clustered_filename               = f'.\\data\\pesentation\\clustered\\{cutout_image}_clustered.png'     
    cv2.imwrite(clustered_filename, segmented_image)
    if is_apple == True:
        clustered_filename_figure               = f'./data/pesentation/clustered/{cutout_image}_clustered_fig.png'
    else:
        clustered_filename_figure               = f'.\\data/pesentation\\clustered\\{cutout_image}_clustered_fig.png'    
    
    st.image(clustered_filename_figure)
    plt.savefig(clustered_filename_figure)
    ############################ END Refinement of KMEAN via KNN ###################################
    ################################################################################################
    # visualizer.fit(pixels) # Fit the data to the visualizer
    # visualizer.show() # Finalize and render the figure
    # plt.show()
    # # Reshape the labels to match the original image shape
    # clustered_image = labels.reshape(height, width) 
    # # Create a visualization of the clustered image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(clustered_image,  cmap = cmap, norm=norm)
    # plt.title('K-means Clustering Result')
    # plt.axis('off')
    #################################### return ####################################################
    ################################################################################################
    return clustered_filename, clustered_filename_figure                                                                          # retunrn of the cluster filenname to get later images via name and not anymore via index, because these are produced images
    ################################END of kmeans_photo_generation##################################

####################################################################################################
########################## Surface calculation function ############################################
####################################################################################################
def surface_calculations(clustered_filename, selected_class):
    """
    gets the clustered image and the class name picked by the user and returns the calculations
    """
    ############################ Extracting image data for estimations #############################
    ################################################################################################
    #clustered_image             = clustered_filename#cv2.imread(clustered_filename)                 # opening image file
    # st.write(np.max(clustered_filename))
    # st.write(np.min(clustered_filename))
    height, width               = clustered_filename.shape  
    # st.write('unique values')                                         # getting dimenesions of image
    # st.write(clustered_filename)
    total_pixels                = height*width                                                       # calculating total pixel area                                
    picked_class_pixels         = np.count_nonzero(clustered_filename == 1)             # calculating picked area pixels
    class_zero                  = np.count_nonzero(clustered_filename==0)   
    

    # st.write('total_pixels')                                         # getting dimenesions of image
    # st.write(total_pixels)


    # st.write('picked_class_pixels')                                         # getting dimenesions of image
    # st.write(picked_class_pixels)

    # st.write('class_zero')                                         # getting dimenesions of image
    # st.write(class_zero)

    ############################ Starting calculation ##############################################
    ################################################################################################
    # percentage_applicable       = round(picked_class_pixels/(class_zero) * 100, 2)    # calculating percentage
  
    estmiation_of_size          = round(picked_class_pixels * 0.0089, 0)                              # recalculating into m2
    number_of_panels            = round(estmiation_of_size/(1.65*0.99) * 0.60, 0)                 # estimating number of panles with buffer of .6 empiricale estimated
    number_of_planels_realistic = round(number_of_panels * 0.60, 0)
    ############################ Estimating energy production ######################################
    ################################################################################################
    low_band_energy             = round(250 * number_of_planels_realistic /1000, 0)                                             # 250 kwh to 300 kwh
    high_band_energy            = round(350 * number_of_planels_realistic /1000, 0)
    average_energy = 29

    st.markdown('________________________________________________________________')
    st.markdown('### Solar Energy Production Report')

    # if average_energy < low_band_energy:
    #     energy_message = f"An average household consumes 29 Kwh per day. This property can potentially produce between {low_band_energy}Kwh and \
    #         {high_band_energy}Kwh which means it can be fully sustainable."
    # elif average_energy > high_band_energy: 
    #     energy_message = f"An average household consumes 29 Kwh per day. This property can potentially produce between {low_band_energy}Kwh and \
    #         {high_band_energy}Kwh and around {average_energy - high_band_energy} Kwh of its needed power has to be provided by the urban electricity network." 
    # else:
    #     energy_message = f"An average household consumes 29 Kwh per day. This property can potentially produce between {low_band_energy}Kwh and \
    #         {high_band_energy}Kwh and has a {round((average_energy - low_band_energy)/(high_band_energy - low_band_energy)*100, 2)}% chance to be fully sustainable"       
    
    energy_message = f'Based on the calculations, the solar panels can potentially save between {round(low_band_energy * 0.14,0)}€ and {round(high_band_energy *0.14, 0)}€ per hour.'
    
    ############################ Genereating Report ################################################
    ################################################################################################
    # report_msg                  = f'The applicable area is percentage_applicable% of the total area which is approximately {estmiation_of_size} sqm. Based on our\
    #       understanding we might be able to install at lest {number_of_panels} pieces of solar panels sized 1.65m x 0.99m which can produce\
    #         {low_band_energy} - {high_band_energy} watts of electricty per hour.'
    # st.write(report_msg)

    ############################ Final report creation #############################################
    ################################################################################################
    def scorecard(title, value, color):
        st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 10px;">'
                    f'<h5 style="color: white;">{title}</h5>'
                    f'<h5 style="color: white;">{value}</h5>'
                    f'</div><br>', unsafe_allow_html=True)

        # Define your metrics
    metric1_title = "Estimated Applicable Area"
    metric1_value = f'{estmiation_of_size}sqm'
    metric1_color = "green"

    metric2_title = "Approximate Number of Installable Panels"
    metric2_value = f' {number_of_planels_realistic} panels'
    metric2_color = "green"

    metric3_title = "Energy Generation(daily)"
    metric3_value = f"Between {low_band_energy} Kwh and {high_band_energy} Kwh"
    metric3_color = "green"


    st.markdown(f'<div style="background-color: #e1e1e6; padding: 10px; border-radius: 10px;">'
                    f'<p style="padding:10px">{energy_message}</p>'
                    f'</div><br><br>', unsafe_allow_html=True)
    

    
    # Create and display the scorecards
    st.write("#### Statistics:")
    scorecard(metric1_title, metric1_value, metric1_color)
    scorecard(metric2_title, metric2_value, metric2_color)
    scorecard(metric3_title, metric3_value, metric3_color)

    #################################### return ####################################################
    ################################################################################################
    # return report_msg
    ################################END of durface calculation #####################################

####################################################################################################    
############################### START Cropped Images ###############################################
####################################################################################################
@st.cache_data(persist="disk")
def generate_filtered_image(clustered_image, class_no):
    #clustered image filtered
    # st.write('This the max value of the saved clustered image')
    # st.write(np.max(clustered_image))
    # testimage = clustered_image*51
    # st.image(testimage)
    clustered_image_filtered            = (clustered_image == class_no).astype(int)
    st.image(clustered_image_filtered*51)

    # print(clustered_image_filtered)
    # Reshape the labels to match the original image shape
    cmap                                = mcolors.ListedColormap(colors=['red'])
    bounds                              = np.arange(np.min(clustered_image_filtered), 
                                                    np.max(clustered_image_filtered) + 1, 1)  # Define class boundaries
    norm                                = mcolors.BoundaryNorm(boundaries=bounds, 
                                                               ncolors=cmap.N, clip=False)
    # cbar                                = plt.colorbar()

    # Create a visualization of the clustered image filtered
    plt.figure(figsize=(8, 8))
    plt.imshow(clustered_image_filtered*51,  cmap = cmap, norm=norm)
    plt.title('K-means Clustering Result')
    plt.axis('off')
    plt.show()
    if is_apple == True:
        filtered_filename                   = f'./data/clustered/_filtered.png'
    else:
        filtered_filename                   = f'.\\data\\clustered\\_filtered.png'    
    plt.savefig(filtered_filename)    
    # plt.figure(figsize=(8, 8))
    # plt.imshow(clustered_image,  cmap = cmap, norm=norm)
    # cbar = plt.colorbar()

    # plt.title('K-means Clustering Result')
    # plt.axis('off')

    # clustered_filename = f'./data/pesentation/clustered/{cutout_image}_clustered.png' 
    # plt.savefig(clustered_filename)
    return filtered_filename, clustered_image_filtered
    ############################### END Cropped Images ##################################################

##############  START find clustered images in a directory###############################################      
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
##############  END find clustered images in a directory #################################################      
#######################################################################################################################
#################################### END OF FUNCTIONS #################################################################
#######################################################################################################################

# generate the kmeans clustering image
cutout_image                            = cutout_files[image_index]
# print(cutout_image)
# plt.figure(figsize=(8, 8))
if st.checkbox('Make a Prediction'):
    clustered_file, clustered_filename_figure  = kmeans_photo_generation(cutout_np, cutout_image, num_clusters=5, colors=COLORS)
    # st.write(clustered_file)
    #cv2.imread(clustered_file)*51
    # st.image(clustered_filename_figure, use_column_width=True)

# Specify the folder path
# clustered_path = "../data/pesentation/clustered"
# all_clustered = list_files_in_directory(clustered_path)
# create and display the kmeans clustered image
# selected_clustered = all_clustered[image_index]
# print(selected_clustered)   
# clsutered_path = os.path.join(clustered_path, selected_clustered)
# st.image(clustered_file, use_column_width=True)

#######################################################################################################################
left_column.markdown("#### Clustering Outcome")
st.markdown("""
### Pick the Surface You Want to Apply the Panels to\n
__1.__ Each color is related to a surface. By looking at the colored bar next to the above photo, pick the surface number. 
            
""")
#######################################################################################################################
st.markdown("________________________________________________________________")
#######################################################################################################################
######################### drop down for the class selection############################################################
#######################################################################################################################
selected_class                              = st.selectbox(label="Pick a surface:", options=[1,2,3,4,5])
#st.write(type(selected_class))
#st.write(selected_class)
######################### generating cropped image for area analysis ##################################################
if st.checkbox('Submit the Selection'):
    clustered_np                            = cv2.imread(clustered_file, cv2.IMREAD_GRAYSCALE)
    #st.write(np.max(clustered_np))
    #st.write(clustered_file)
    filtered_filename, clustered_image_filtered = generate_filtered_image(clustered_np, selected_class)# plt.figure(figsize=(8, 8))
    #st.write(clustered_image_filtered)
    #st.image(clustered_file, use_column_width=True)
    #filtered_clustered_np                    = cv2.imread(clustered_image_filtered)
    if st.checkbox('Generate the Final Report'):
        report                       = surface_calculations(clustered_image_filtered, selected_class) #generate_filtered_image(clustered_np, selected_class)
        # st.write(filtered_file )
        #st.image(report, use_column_width=True)