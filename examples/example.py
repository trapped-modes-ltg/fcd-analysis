import sys; sys.path.insert(0, '..')
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD
import os

displaced_image = Image("./example_images/sessile_drop.png", roi=-1)
reference_image = Image("./example_images/reference.png", roi=displaced_image.roi)

layers = [Layer(5e-3, "Air"), Layer(1e-3, "Glass"), Layer(0, "Distilled_water"), Layer(1, "Air")]
fcd = FCD(reference_image.windowed(), layers=layers, square_size=1e-3)
height_field = fcd.analyze(displaced_image.windowed())
height_field.show_slice()
