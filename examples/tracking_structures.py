import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydata.video import Video
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD

base_dir = os.path.dirname(__file__)
displaced_images_path = os.path.join(base_dir, "floating_structure")
template_image_path = os.path.join(base_dir, "static_images", "structure.png")

template = Image(template_image_path, roi=-1)
template.center_toroid()
template.rotated(angle=45)

video = Video(displaced_images_path, start_frame=0)
    
def pre_process(frame):
    frame.track(template.processed)
    return frame.processed
video.show_movie(pre_process=pre_process)
