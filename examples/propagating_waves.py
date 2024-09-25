import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydata.video import Video
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD

base_dir = os.path.dirname(__file__)
displaced_images_path = os.path.join(base_dir, "propagating_waves")
reference_image_path = os.path.join(base_dir, "static_images", "reference_2.png")

video = Video(displaced_images_path)
roi = video.current_frame.select_roi()
reference_image = Image(reference_image_path, roi=roi)

fcd = FCD(reference_image.windowed(), square_size=0.0022, height=0.0323625)
def apply_fcd(frame):
    height_field = fcd.analyze(frame.windowed(), full_output=False)
    return height_field
video.show_movie(pre_process=apply_fcd)
