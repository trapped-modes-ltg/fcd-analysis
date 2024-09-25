import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydata.video import Video
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD

base_dir = os.path.dirname(__file__)
displaced_images_path = os.path.join(base_dir, "floating_structure")
reference_image_path = os.path.join(base_dir, "static_images", "reference_2.png")
template_image_path = os.path.join(base_dir, "static_images", "structure.png")

template = Image(template_image_path, roi=-1)
template.center_toroid()
template.rotated(angle=45)

c = template.roi[2] // 2
R = int(45 / 1.41)
reference = Image(reference_image_path, roi=template.roi)
layers = [Layer(5.7e-2, "Air"), Layer(1.2e-2, "Acrylic"), Layer(4.3e-2, "Distilled_water"), Layer(80e-2, "Air")]
fcd = FCD(reference.windowed()[c-R:c+R, c-R:c+R], square_size=0.0022, layers=layers)
mask = reference.make_circular_mask(radius=45)

video = Video(displaced_images_path, start_frame=0)

def pre_process(frame):
    frame.track(template.processed)
    reference.roi = frame.roi  # Ir desplazando también el reference.
    frame.masked(mask, reference.windowed())
    height_field = fcd.analyze(frame.windowed()[c-R:c+R, c-R:c+R], full_output=False)
    return height_field
video.show_movie(pre_process=pre_process, end_frame=12)
