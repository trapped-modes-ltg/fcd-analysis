import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyfcd.layer import Layer
from pydata.image import Image
from pyfcd.fcd import FCD

base_dir = os.path.dirname(__file__)
displaced_image_path = os.path.join(base_dir, "static_images", "sessile_drop.png")
reference_image_path = os.path.join(base_dir, "static_images", "reference_1.png")

displaced_image = Image(displaced_image_path, roi=-1)
reference_image = Image(reference_image_path, roi=displaced_image.roi)

layers = [Layer(5e-3, "Air"), Layer(1e-3, "Glass"), Layer(0, "Distilled_water"), Layer(1, "Air")]
fcd = FCD(reference_image.windowed(), layers=layers, square_size=1e-3)
height_field = fcd.analyze(displaced_image.windowed())
height_field.show_slice()
