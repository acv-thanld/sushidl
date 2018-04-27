import os
from os import listdir
from PIL import Image

def rotate_image(img = '', deg = 1, dir = ''):
	if not img: 
		return
	print(deg)
	im = Image.open(img)
	img_new = im.rotate(deg, expand=True)
	img_new.save(dir)

def read_dir(dir = ""):
	return listdir(dir)

def check_dir(dir = ""):
	if not dir: return
	if not os.path.exists(dir):
		os.makedirs(dir)
	return True

def convert_array_to_dict(arr = []):
	if not arr: return False
	dict = {}
	for i,v in enumerate(arr):
		if(v == '.picasa.ini'):
			continue
		dict.update({str(i + 1):v})

	return dict