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




# all_addresses = []

# # [("Alt-Lankwitz", "12247", 1, 200), ("Ritterstraße","10969", 1, 125)]

# initial_addresses = pd.read_csv("../data/initial_addresses.csv", header=0)
# initial_addresses = [("Alt-Lankwitz", "12247", 1, 200)]

# for street, plz, low, high in initial_addresses:
#     generated = generate_address(street, plz, low, high)
#     all_addresses += generated

# print(all_addresses)







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


# df = get_lat_lon(all_addresses)

















