from google.cloud import storage
from io import BytesIO, StringIO
from PIL import Image
import numpy as np

json_keyfile_path = '../galvanic-fort-399815-b650c8e36bea.json'
bucket_name = 'datasolamente'
blob_name = 'data/Splittingfolder_temp/104_Ritterstrasse/104_Ritterstraße_Hyp.png'




def read_image_from_gcs(bucket_name, blob_name, keyfile):
    """
    Download an image from Google Cloud Storage and return it as a NumPy array.
    """
    try:
        # Initialize the Google Cloud Storage client with the service account JSON key.
        client = storage.Client.from_service_account_json(keyfile)

        # Get a reference to the specified bucket and blob.
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download the image as bytes.
        image_bytes = blob.download_as_bytes()

        # Convert the bytes to a PIL Image.
        pil_image = Image.open(BytesIO(image_bytes))

        # Convert the PIL Image to a NumPy array.
        numpy_array = np.array(pil_image)

        return numpy_array
    except Exception as e:
        print(f"Error downloading image from GCS: {e}")
        return None

    

def read_text_file_from_gcs(bucket_name, file_name, json_key_path):
    """
    Read a text file from Google Cloud Storage using a service account JSON key.
    """
    try:
        # Initialize the Google Cloud Storage client with the service account JSON key.
        client = storage.Client.from_service_account_json(json_key_path)

        # Get a reference to the specified bucket and file.
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Download the content of the text file.
        text_content = blob.download_as_text()

        return text_content
    except Exception as e:
        print(f"Error reading text file from GCS: {e}")
        return None





from google.cloud import storage
import pandas as pd

def download_csv_as_dataframe(bucket_name, blob_name, json_key_path):
    """
    Download a CSV file from Google Cloud Storage and return it as a Pandas DataFrame.
    """
    try:
        # Initialize the Google Cloud Storage client with the service account JSON key.
        client = storage.Client.from_service_account_json(json_key_path)

        # Get a reference to the specified bucket and blob.
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download the CSV file as a string.
        csv_data = blob.download_as_text()

        # Convert the CSV data to a Pandas DataFrame.
        df = pd.read_csv(StringIO(csv_data))

        return df
    except Exception as e:
        print(f"Error downloading CSV file and creating DataFrame: {e}")
        return None

























def write_dataframe_to_gcs(dataframe, bucket_name, blob_name, json_key_path):
    """
        upload a df to  GCS
    """
    try:
        # Initialize the Google Cloud Storage client with the service account JSON key.
        client = storage.Client.from_service_account_json(json_key_path)

        # Get a reference to the specified bucket.
        bucket = client.bucket(bucket_name)

        # Create a BytesIO buffer to store the DataFrame as CSV data.
        csv_buffer = BytesIO()
        dataframe.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_data = csv_buffer.getvalue()

        # Upload the CSV data to the specified blob in GCS.
        blob = bucket.blob(blob_name)
        blob.upload_from_string(csv_data, content_type='text/csv')

        print(f"the dataframe {dataframe.name} successfully uploaded to GCS!")
        return True
    except Exception as e:
        print(f"Error writing DataFrame to GCS: {e}")
        return False










def write_bytes_to_gcs_as_image(bucket_name, blob_name, image_bytes, json_key_path):
    """
    Write image bytes to Google Cloud Storage as an image file using a service account JSON key.
    """
    try:
        # Initialize the Google Cloud Storage client with the service account JSON key.
        client = storage.Client.from_service_account_json(json_key_path)

        # Get a reference to the specified bucket and create the blob (image) with the given content.
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Upload the image bytes to the specified blob in GCS.
        blob.upload_from_string(image_bytes, content_type="image/jpeg")  # Adjust content_type if needed.

        return True
    except Exception as e:
        print(f"Error writing image to GCS: {e}")
        return False




# Example usage:
if __name__ == "__main__":
    # Replace these values with your own.
    blob_name = "data/sample_iamges/map_image.png"

    # Read the image as bytes (for example, from a local file).
    with open("path/to/your/local/image.jpg", "rb") as f:
        image_data = f.read()

    # Write the image bytes to Google Cloud Storage.
    success = write_bytes_to_gcs_as_image(bucket_name, blob_name, image_data, json_key_path)

    if success:
        print("Image written to GCS successfully.")




