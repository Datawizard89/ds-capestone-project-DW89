# from gcs_integration import read_text_file_from_gcs, download_csv_as_dataframe, write_bytes_to_gcs_as_image
import gcs_integration as gcs
import requests
import pandas as pd
import os 
import numpy as np



api_file = "data/StaticKey.txt"
json_keyfile_path = '../galvanic-fort-399815-b650c8e36bea.json'
bucket_name = 'datasolamente'




google_map_static_api_key = gcs.read_text_file_from_gcs(bucket_name, api_file, json_keyfile_path)


addresses_df = gcs.download_csv_as_dataframe(bucket_name, "data/address_lat_long.csv", json_keyfile_path)
BASE_URL = "https://maps.googleapis.com/maps/api/staticmap?"

print(addresses_df.head())

def get_google_images(addresses_df):
    addresses_df['center'] = addresses_df['address'].str.replace(' ', '+')
    center_list = addresses_df.center.to_list()

    for item in center_list:
        satelite_img_url = BASE_URL + "center=" + item + "&zoom=" + \
            str(20) + "&size=400x400&maptype=satellite&key=" + google_map_static_api_key
        map_img_url = BASE_URL + "center=" + item + "&zoom=" + \
            str(20) + "&size=400x400&maptype=roadmap&key=" + google_map_static_api_key

        satelite_img = requests.get(satelite_img_url)    
        map_img = requests.get(map_img_url) 
        
        return map_img.content
        # print(type(map_img.content))

        break
        

sample_img = get_google_images(addresses_df)   

blob_name = "data/sample_images/map_image.png"

gcs.write_bytes_to_gcs_as_image(bucket_name, blob_name, sample_img, json_keyfile_path)

