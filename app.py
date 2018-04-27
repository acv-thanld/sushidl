import flask
import time
import json
import sys
import io
import glob, os, math
import requests
from PIL import Image
from flask import Flask, render_template, url_for, send_from_directory
import skimage.io
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from PIL import Image

import urllib.request


from __init__ import *
from __func__ import *
from os.path import isfile, join

import mrcnn.utils
import mrcnn.model as modellib
from mrcnn import visualize
import sushi

model = None
app = init()

@app.route('/findsushi')
def findsushi():
        # Root directory of the project
        ROOT_DIR = os.path.abspath("/media/than/3E7019A6701965C5/nhandt_MaskRCNN_Nouth/")
        
        
        
        
        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        
        # Import COCO config
        sys.path.append(os.path.join(ROOT_DIR, "samples/sushi/"))  # To find local version
        
        
        
       
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        
        image_type = flask.request.args.get("type") # load
        #arr_out = { "north": "", "south": "nhandt_MaskRCNN_South/", "east": "nhandt_MaskRCNN_East/", "west": "nhandt_MaskRCNN_West/" }
        # Local path to trained weights file
        #COCO_MODEL_PATH = os.path.join(ROOT_DIR, arr_out.get(image_type) + "mask_rcnn_sushi_0030.h5")    
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "nhandt_MaskRCNN_West/mask_rcnn_sushi_0030.h5")   
        #print(COCO_MODEL_PATH)
        #return ''
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
                utils.download_trained_weights(COCO_MODEL_PATH)
        
        
        # Directory of images to run detection on
        IMAGE_DIR = os.path.join(ROOT_DIR, "images")


        class InferenceConfig(sushi.SushiConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()


        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)


        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        #class_names = ['BG', 'IKR', 'MGR', 'HMC', 'ANG', 'Shg',
        #       'Ika', 'Ebi', 'SM', 'TMG', 'HIKR',
        #       'NGTR', 'Oke']
        
        class_names = ['BG', 'TBK', 'MGR', 'TKM', 'ANG', 'ShG',
               'Ika', 'Ebi', 'SaM', 'TMG', 'HIKR',
               'NGTR', 'Oke']

        # Load a random image from the images folder
       # file_names = next(os.walk('/media/than/3E7019A6701965C5/nhandt_MaskRCNN_Nouth/datasets/sushi/test/'))[2]
        image_id = flask.request.args.get("id")
        
        
        #

                
        image_url = flask.request.args.get("image")
        area = image_type + '/' + image_id + '/'
        directory = '/media/than/3E7019A6701965C5/nhandt_MaskRCNN_Nouth/output/' + area
        if not os.path.exists(directory):
                os.makedirs(directory)
        file_names = directory + 'INPUT.JPG'
        
        # create a password manager
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

        # Add the username and password.
        # If we knew the realm, we could use it instead of None.
        top_level_url = "http://52.198.65.54/"
        password_mgr.add_password(None, top_level_url, 'sushi', 'sushi!!')

        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

        # create "opener" (OpenerDirector instance)
        opener = urllib.request.build_opener(handler)

        # use the opener to fetch a URL
        #opener.open(a_url)

        # Install the opener.
        # Now all calls to urllib.request.urlopen use our opener.
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(image_url, file_names)
        
 

        #visualize.save_image(image, file_names, r['rois'], r['masks'], r['class_ids'], r['scores'], class_names, filter_classs_names=None, scores_thresh=0.1, save_dir='/media/than/3E7019A6701965C5//nhandt_MaskRCNN_Nouth', mode=0)
        image = skimage.io.imread(os.path.join(directory, file_names))



        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]

        #visualize.save_image(image, file_names, r['rois'], r['masks'], r['class_ids'], r['scores'], class_names, filter_classs_names=None, scores_thresh=0.1, save_dir='/media/than/3E7019A6701965C5/ nhandt_MaskRCNN_Nouth', mode=0)
        
        visualize.save_image(image, [file_names], r['rois'], r['masks'], r['class_ids'], r['scores'], class_names, filter_classs_names=None, scores_thresh=0.1, save_dir=directory, mode=0)
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            #class_names, r['scores'])
        
        
        #demo_sushi()
        return flask.jsonify({ "status": "ok", "path_out": "/images/" + area + 'out.JPG', "id": image_id })
        #return flask.jsonify({ "status": "ok", "path_out": "http://172.31.13.127:5000/images/" + area + 'out.JPG', "id": image_id}) 


@app.route('/')
def index():
###
#	DS = str(os.path.sep)
#	ROOT = DS.join(app.instance_path.split(DS)[:-1])
#	path_out = ''
#	subfix_image_output = '_' + str(time.time()).split('.')[0]
#	url = ''
#	data_out = {"status":"queue"}
#	arr_out = { "north": "/nhandt_MaskRCNN_Nouth/", "south": "/nhandt_MaskRCNN_South/", "east": "/nhandt_MaskRCNN_East/", "west": "/nhandt_MaskRCNN_West/" }
#	arr_input = { "north": "python3 /media/than/3E7019A6701965C5/nhandt_MaskRCNN_Nouth/demo_sushi.py", "south": "python3 /media/than/3E7019A6701965C5/nhandt_MaskRCNN_South/demo_sushi.py", "east": "python3 /media/than/3E7019A6701965C5/nhandt_MaskRCNN_South/demo_sushi.py", "west": "python3 /media/than/3E7019A6701965C5/nhandt_MaskRCNN_South/demo_sushi.py)" }

#	if flask.request.args.get("type"):
#		filename = flask.request.args.get("image").split("/")[-1]
#		arr_filename = filename.split(".")
#		arr_filename[0] = arr_filename[0] + subfix_image_output
#		filename = ".".join(arr_filename)
#
#		image = requests.get(flask.request.args.get("image")).content
#		image = Image.open(io.BytesIO(image))
#		path_out = arr_out[flask.request.args.get("type")] + filename
#		image.save(ROOT + path_out)

#		if arr_input[flask.request.args.get("type")] != None:
#			print(arr_input.get(flask.request.args.get("type")))
#			#os.system(arr_input.get(flask.request.args.get("type")) +" "+ flask.request.args.get("image") + ' ' + filename)

###
#	return flask.jsonify({ "status": "ok", "path_out": url_for("send_image",domain=flask.request.args.get("type"),file=filename) })
        return ""
@app.route('/gallery')
def get_gallery():
	#f = open('images/' + str(pid) + '.jpg', 'r+')
	image_names = os.listdir('images')
	return render_template("gallery.html", image_names=image_names)

# domain / filename.   => http://127.0.0.1:5000/sushi/east/4.jpg
@app.route('/images/<img_type>/<img_id>/<img_file>')
def send_image(img_type, img_id, img_file):
	#DS = str(os.path.sep)
	#ROOT = DS.join(app.instance_path.split(DS)[:-1])
	#arr_out = { "north": "/output/north/", "south": "/output/south/", "east": "/output/east/", "west": "/output/west/" }
	#file_image = arr_out[domain] + file
	#return send_from_directory("./output/" , file)
        area = img_type + '/' + img_id + '/'
        directory = '/media/than/3E7019A6701965C5/nhandt_MaskRCNN_Nouth/output/' + area
        return send_from_directory(directory, img_file)

if __name__ == '__main__':
	# app.run(DEBUG=True)
	app.run("",port=5000)
