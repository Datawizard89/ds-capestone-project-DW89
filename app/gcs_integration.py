from google.cloud import storage
from io import BytesIO
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








