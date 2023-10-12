from geopy.geocoders import Nominatim
import pandas as pd
import os
import time


def generate_address(street_name, plz, low, high):
    """
    given a street name and a plz, and range of houses from 1-200, generate an address
    """
    address_urb = []
    
    try:
        for item in range(low, high):
            address_urb.append(str(item) +" "+ street_name +" "+ plz+ " " + "Berlin")

        print(f'{str(high-low)} addresses for street name {street_name} and plz {plz} generated!')
        return address_urb

    except:
        print('there is a problem fetching addresses!')





def get_lat_lon(address_list):
    """
    get lat and long given address
    """
    geolocator = Nominatim(user_agent="http")

    lat_list = []
    long_list = []

    try:
        for item in range(len(address_list)):
            location = geolocator.geocode(address_list[item])
            lat_list.append(location.latitude)
            long_list.append(location.longitude)

    
        df = pd.DataFrame({
            'address' : address_list, 
            'lat' : lat_list,
            'long' : long_list	 
         })
        
        print(f"Latitude and Longitude for {len(address_list)} addresses generated!")
    except:
        print(f"There is a problem generating the latitude and longitudes!")
   


    return df














