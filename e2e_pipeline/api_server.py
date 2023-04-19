'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Ning Yu
'''

import html
import json
import os
import sys
from io import BytesIO
from urllib.parse import urljoin

from PIL import Image
from flask import Flask, flash, request, url_for, send_from_directory
from flask import jsonify
from flask_cors import CORS
from selenium import webdriver
from werkzeug.utils import secure_filename

from gen_single_sample_API_server import generate_banners, load_model
from utils_server import safeMakeDirs, generate_htmls
from selenium import webdriver
from selenium.webdriver import Chrome


sys.path.append('../')

# set up uploading params
UPLOAD_FOLDER = 'uploaded_images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
STYLE_FOLDER = 'style_jsons'
GENERATED_FOLDER = 'generated_images'
MODEL_FOLDER = '/export/home/model/content_generation'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)
app.add_url_rule(
    "/generated/<name>", endpoint="generated_files", build_only=True
)
app.root_path = '/home/Claude'

# Uncomment this line if you are making a Cross domain request
CORS(app)

# model dictionary
loaded_models = {}

browser = None


# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_url_by_basename(host_url, base_name):
    return urljoin(host_url, url_for('generated_files', name=base_name)) if base_name != "" else ""


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route('/generated/<name>')
def get_generated_file(name):
    return send_from_directory(app.config["GENERATED_FOLDER"], name)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    safeMakeDirs(app.config['UPLOAD_FOLDER'])
    host_url = request.host_url
    res = {'status': 'fail',
           'url': ''}
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No image part')
            return res
        file = request.files['image']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return res
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            url = urljoin(host_url, url_for('download_file', name=filename))
            res = {'status': 'success',
                   'url': url}
            return json.dumps(res)
    return res


@app.route('/prediction', methods=['POST'])
def content_generation():
    host_url = request.host_url
    request_dict = request.json
    content_style = request_dict['contentStyle']
    global browser

    background_image_url = content_style['backgroundImage']
    # load the image directly if it is hosted on the same server
    background_image_path = ''
    if host_url in background_image_url:
        image_basename = os.path.basename(background_image_url)
        background_image_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], image_basename)
    else:
        # TODO: add functions to download images from url
        print(f'Downloading image from {background_image_url}')

    # construct the strings
    input_string = ''
    for ele in content_style['elements']:
        try:
            if ele['type'] == 'header' or ele['type'] == 'body' or ele['type'] == 'button':
                input_string += f'{ele["text"]}|'
        except KeyError:
            continue
    if len(input_string) > 0:
        input_string = input_string[:-1]  # remove tailing '|'

    print(f'input string: {input_string}')

    model_id = request_dict['modelId']
    if model_id not in loaded_models:
        print(f'Loading model {model_id}')
        lo_model = load_model(os.path.join(app.config['MODEL_FOLDER'], model_id))
        loaded_models[model_id] = lo_model
        print(f'Loaded model {model_id}')

    if not browser:
        # initialize Chrome based web driver for html screenshot
        options = webdriver.ChromeOptions()
        options.add_argument('no-sandbox')
        options.add_argument('headless')
        browser = Chrome(executable_path='/usr/bin/chromedriver', options=options)
        browser.set_window_size(1500, 1500)

    # generate images
    out_dir = os.path.join(app.root_path, app.config['GENERATED_FOLDER'])
    safeMakeDirs(out_dir)

    seeds = [x + 1 for x in range(int(request_dict['numResults']))]
    # generate banner ads variation with the pre-defined post-processing probabilities
    generated_image_paths, generated_html_paths = generate_banners(loaded_models[model_id], background_image_path,
                                                                   content_style['elements'],
                                                                   {'jitter': 5/6,
                                                                    'horizontal_center_aligned': 2/3,
                                                                    'horizontal_left_aligned': 1/3,
                                                                    },
                                                                   seeds,
                                                                   request_dict['resultFormat'], browser, out_dir)

    res = {'generatedResults': [],
           'task': request_dict['task']}

    for gen_image_path, gen_html_path in zip(generated_image_paths, generated_html_paths):
        gen_image_basename = os.path.basename(gen_image_path)
        gen_image_url = get_url_by_basename(host_url, gen_image_basename)
        gen_html_basename = os.path.basename(gen_html_path)
        gen_html_url = get_url_by_basename(host_url, gen_html_basename)
        res['generatedResults'].append({
            'image_url': gen_image_url,
            'html_url': gen_html_url
        })
    print(res)
    return jsonify(res)


@app.route('/update', methods=['POST'])
def update_files():
    """
    update the generated image / html using the posted payload
    """
    request_dict = request.json
    edited_htmls = request_dict['editedHTMLs']
    response = {
        "updatedStatus": []
    }
    w_page, h_page = 600, 400  # thumbnail resolution
    global browser
    if not browser:
        # initialize Chrome based web driver for html screenshot
        options = webdriver.ChromeOptions()
        options.add_argument('no-sandbox')
        options.add_argument('headless')
        browser = Chrome(executable_path='/usr/bin/chromedriver', options=options)
        browser.set_window_size(1500, 1500)

    for item in edited_htmls:
        html_name = item['htmlName']
        # html_encoded = item['htmlContent']
        # html_decoded = html.unescape(html_encoded)
        html_decoded = item['htmlContent']
        # save
        cur_status = "success"
        try:
            html_path = os.path.join(app.root_path, app.config["GENERATED_FOLDER"], html_name)
            with open(html_path, "w") as file:
                file.write(html_decoded)

            # update image
            html_dir_and_name, _ = os.path.splitext(html_path)
            image_path_vis = f'{html_dir_and_name}_vis.png'
            image_path_ori = f'{html_dir_and_name}.png'
            original_image = Image.open(image_path_ori)
            W_page, H_page = original_image.size
            # TODO: refactor this code because it's similar to the one in gen_single_sample_API_server.py
            try:
                browser.get("file:///" + html_path)
            except Exception as e:
                pass
            png = browser.get_screenshot_as_png()
            screenshot = Image.open(BytesIO(png))
            screenshot = screenshot.crop([0, 0, W_page, H_page])
            if W_page > w_page or H_page > h_page:
                screenshot.thumbnail((w_page, h_page), Image.ANTIALIAS)
            screenshot.save(image_path_vis)

        except Exception as e:
            print(e)
            cur_status = "error"
        response['updatedStatus'].append({
            "htmlName": html_name,
            "status": cur_status
        })
    return jsonify(response)


@app.route('/save', methods=['GET', 'POST'])
def save_file():
    """
    this is a dummpy endpoint to support a save endpoint
    """
    res = {"status": "success"}
    return jsonify(res)