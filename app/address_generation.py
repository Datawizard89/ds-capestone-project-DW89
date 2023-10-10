from geopy.geocoders import Nominatim
import pandas as pd
import os


def generate_address(street_name, plz, low, high):
    """
    given a street name and a plz, and range of houses from 1-200, generate an address
    """
    address_urb = []
    
    for item in range(low, high):
        address_urb.append(str(item) +" "+ street_name +" "+ plz+ " " + "Berlin")

    return address_urb

all_addresses = []
for street, plz, low, high in [("Alt-Lankwitz", "12247", 1, 200), ("Ritterstraße","10969",1,125)]:
    generated = generate_address(street, plz, low, high)
    all_addresses += generated


print(len(all_addresses))
