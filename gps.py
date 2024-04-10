# Write a code to integrate neo6m gps module with raspberry pi.
# The code should take input from the Image processing model and display the location of the dead body on the map.
# The code should also display the location of the user on the map.
# The code should also display the distance between the user and the human body on the map.
# The code should also display the direction of the human body from the user on the map.

import serial
import pynmea2
import folium
import webbrowser
import geopy.distance
import math
import time

# Function to get the current location of the user
def get_current_location():
    port = "/dev/ttyAMA0"
    ser = serial.Serial(port, baudrate=9600, timeout=0.5)
    while True:
        try:
            data = ser.readline().decode('ascii', errors='replace')
            if data[0:6] == '$GPGGA':
                msg = pynmea2.parse(data)
                latitude = msg.latitude
                longitude = msg.longitude
                return latitude, longitude
        except:
            pass

# Function to get the location of the dead body
def get_dead_body_location():
    latitude = 12.9716
    longitude = 77.5946
    return latitude, longitude

# Function to calculate the distance between the user and the dead body
def calculate_distance(user_latitude, user_longitude, dead_body_latitude, dead_body_longitude):
    user_location = (user_latitude, user_longitude)
    dead_body_location = (dead_body_latitude, dead_body_longitude)
    distance = geopy.distance.distance(user_location, dead_body_location).m
    return distance

# Function to calculate the direction of the dead body from the user
def calculate_direction(user_latitude, user_longitude, dead_body_latitude, dead_body_longitude):
    user_location = (user_latitude, user_longitude)
    dead_body_location = (dead_body_latitude, dead_body_longitude)
    direction = geopy.distance.distance(user_location, dead_body_location).bearing
    return direction

# Function to display the map
def display_map(user_latitude, user_longitude, dead_body_latitude, dead_body_longitude):
    map = folium.Map(location=[user_latitude, user_longitude], zoom_start=15)
    folium.Marker(location=[user_latitude, user_longitude], popup="User").add_to(map)
    folium.Marker(location=[dead_body_latitude, dead_body_longitude], popup="Dead Body").add_to(map)
    folium.PolyLine(locations=[[user_latitude, user_longitude], [dead_body_latitude, dead_body_longitude]], color='red').add_to(map)
    map.save('map.html')
    webbrowser.open('map.html')

# Main function
def main():
    user_latitude, user_longitude = get_current_location()
    dead_body_latitude, dead_body_longitude = get_dead_body_location()
    distance = calculate_distance(user_latitude, user_longitude, dead_body_latitude, dead_body_longitude)
    direction = calculate_direction(user_latitude, user_longitude, dead_body_latitude, dead_body_longitude)
    display_map(user_latitude, user_longitude, dead_body_latitude, dead_body_longitude)
    print("User Location: ", user_latitude, user_longitude)
    print("Dead Body Location: ", dead_body_latitude, dead_body_longitude)
    print("Distance: ", distance, "m")
    print("Direction: ", direction, "degrees")

if __name__ == "__main__":
    main()

# End of code