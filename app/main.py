from address_generation import get_lat_lon, generate_address
from gcs_integration import write_dataframe_to_gcs, read_image_from_gcs
from io import BytesIO
import pandas as pd


bucket_name = 'datasolamente'
blob_name = "data/address_lat_long.csv"
json_key_path = '../galvanic-fort-399815-b650c8e36bea.json'

if __name__ == '__main__':

    initial_addresses = [("Alt-Lankwitz", "12247", 1, 200)]

    # generate addresses
    all_addresses = []

    for street, plz, low, high in initial_addresses:
        generated = generate_address(street, plz, low, high)
        all_addresses += generated



    # generate lat, long for each address
    lat_long_df = get_lat_lon(all_addresses)




    # write the lat, long dataset in cloud
    write_dataframe_to_gcs(lat_long_df, bucket_name, blob_name, json_key_path)

