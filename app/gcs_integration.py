from google.cloud import storage


json_keyfile_path = '../galvanic-fort-399815-b650c8e36bea.json'
bucket_name = 'datasolamente'
image_path = 'data/Splittingfolder_temp/104_Ritterstrasse/104_Ritterstraße_Hyp.png'

def read_image_from_gcs(bucket_name, image_path):
    """Read an image from Google Cloud Storage.
    """
    try:
        # Create a client to interact with Google Cloud Storage
        storage_client = storage.Client.from_service_account_json(json_keyfile_path)

        # Get a reference to the bucket
        bucket = storage_client.get_bucket(bucket_name)

        # Get a blob (object) representing the image
        blob = bucket.blob(image_path)

        # Download the image as bytes
        image_data = blob.download_as_bytes()

        return image_data

    except Exception as e:
        raise Exception(f"Error reading image from Google Cloud Storage: {str(e)}")



# Example usage:
if __name__ == "__main__":
    try:
        image_bytes = read_image_from_gcs(bucket_name, image_path)
        # You can now use 'image_bytes' for further processing, such as displaying or saving the image.
    except Exception as e:
        print(f"Error: {str(e)}")


    # write image bytes to file
    with open("sample2.png", "wb") as f:
    
        # Write bytes to file
        f.write(image_bytes)





# def write_data_to_gcs(bucket_name, object_name, data):
#     """Write data to Google Cloud Storage."""
#     try:
#         # Create a client using Application Default Credentials (ADC)
#         storage_client = storage.Client()

#         # Get the bucket and object
#         bucket = storage_client.get_bucket(bucket_name)
#         blob = bucket.blob(object_name)

#         # Upload the data
#         blob.upload_from_string(data)

#         return True

#     except Exception as e:
#         print(f"Error writing data to GCS: {e}")
#         return False








