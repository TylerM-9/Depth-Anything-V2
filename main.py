import cv2
from cv2_enumerate_cameras import enumerate_cameras

print("Searching for connected video devices...")
cameras = list(enumerate_cameras())

if not cameras:
    print("No cameras detected. Make sure the Camo desktop app is running and your phone is connected.")
else:
    for camera in cameras:
        # This will output the index number you need to pass to Depth-Anything
        print(f"Index: {camera.index} | Name: {camera.name}")
