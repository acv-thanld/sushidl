import flask
import os
from flask import Flask, render_template

def init():
	app = flask.Flask(__name__)
	DS = str(os.path.sep)
	ROOT = DS.join(app.instance_path.split(DS)[:-1]) + DS
	config = {
		"path": {
			"root": ROOT
			, "input" : ROOT + "input" + DS
			, "output" : ROOT + "output" + DS
		}
	}
	app.custom = config

	return app

