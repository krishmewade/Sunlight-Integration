import cv2
import numpy as np
from datetime import datetime, timedelta
from scipy.integrate import simps
import math
from skyfield.api import load, Topos
eph = load('de421.bsp')

def detect_sun(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sun_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(sun_contour)

    sun_centroid_x = int(M['m10'] / M['m00'])
    sun_centroid_y = int(M['m01'] / M['m00'])
    az =  get_solar_position(sun_centroid_x, sun_centroid_y)

    sun_elevation_angle = 90 - sun_centroid_y
    sun_azimuth_angle = (sun_centroid_x / img.shape[1]) * 360  
    sun_elevation_angle_rad = math.radians(sun_elevation_angle)
    sun_azimuth_angle_rad = math.radians(sun_azimuth_angle)
    sun_vector = np.array([
        math.cos(sun_azimuth_angle_rad) * math.cos(sun_elevation_angle_rad),
        math.sin(sun_azimuth_angle_rad) * math.cos(sun_elevation_angle_rad),
        math.sin(sun_elevation_angle_rad)
    ])

    return sun_vector


def get_solar_position(location, time):
  ts = load.timescale()
  t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
  earth, sun = eph['earth'], eph['sun']
  observer = earth + Topos(*location)
  astrometric = (observer + sun).at(t)
  apparent = astrometric.apparent()
  return apparent.altaz()

def is_obstructed(image, point):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dilated_edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    for contour in contours:
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            return True  # if something is in the way
    return False  

def calculate_sunlight_duration(image_folder, location_coordinates, start_time, end_time):
    total_sunlight_duration = 0.0
    sun_vectors = []

    current_time = start_time
    while current_time <= end_time:
        image_filename = f"{current_time.strftime('test')}.jpg"
        image_path = f"{image_folder}/{image_filename}"

        #getvector
        sun_vector = detect_sun(image_path)

        #obstruction
        if not is_obstructed(cv2.imread(image_path), sun_vector):
            sun_vectors.append(sun_vector)

        current_time += timedelta(hours=1)

    sun_vectors = np.array(sun_vectors)

    #integrate
    x = np.arange(len(sun_vectors))
    total_sunlight_duration = simps(y=sun_vectors[:, 2], x=x)

    return total_sunlight_duration #run and use this as num hours of detected sunlight 
