from google.cloud import storage

def read_data_from_gcs(bucket_name, object_name):
    """Read data from Google Cloud Storage."""
    try:
        # Create a client using Application Default Credentials (ADC)
        storage_client = storage.Client()

        # Get the bucket and object
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(object_name)

        # Download the data
        data = blob.download_as_text()

        return data

    except Exception as e:
        print(f"Error reading data from GCS: {e}")
        return None

# Example usage
# if __name__ == "__main__":
#     bucket_name = "your-bucket-name"
#     object_name = "path/to/your/object.txt"
    
#     data = read_data_from_gcs(bucket_name, object_name)
    
#     if data:
#         print(f"Data from GCS: {data}")
#     else:
#         print("Failed to read data from GCS.")




def write_data_to_gcs(bucket_name, object_name, data):
    """Write data to Google Cloud Storage."""
    try:
        # Create a client using Application Default Credentials (ADC)
        storage_client = storage.Client()

        # Get the bucket and object
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(object_name)

        # Upload the data
        blob.upload_from_string(data)

        return True

    except Exception as e:
        print(f"Error writing data to GCS: {e}")
        return False








