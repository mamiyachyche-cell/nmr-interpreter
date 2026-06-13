import os
import zipfile
import tempfile
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import nmrglue as ng

app = Flask(__name__)
CORS(app, origins="*")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')
