import matplotlib.pyplot as plotlib
import numpy
import os
from PIL import Image

def img_dir_resize(dir, width, height):
	for filename in os.listdir(dir):
		path = dir + filename
		image = Image.open(path)
		image = image.resize((width, height))
		image.save(path)
		image.close()
		print(path)
		
img_dir_resize("../data/clean/", 400, 300)
img_dir_resize("../data/dirty/", 400, 300)
img_dir_resize("../data/borderline/", 400, 300)
