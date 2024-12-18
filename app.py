import csv
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
import logging

if not app.debug:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Only log errors

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, primary_choice, secondary_choice):
    print(primary_choice,secondary_choice)
    enhancement_functions = {
        'Mathematical': {
            'AdaptiveThreshold': adaptive_thresholding,
            'Sharpening': sharpening,
            'HistogramEqualization': histogram_equalization,
            'GammaCorrection': gamma_correction,
            'RetinexAlgo': retinex
        },
        'Traditional': {
            'DNCNN': dncNN,
            'RedNet': redNet,
            'VDSR': vdsr,
            'EDSR': edsr,
            'SRCNN': srcnn
        },
        'Novelty': {
            'ResNetGan': resnet_gan
        }
    }
    img = cv2.imread(image_path)
    if primary_choice in enhancement_functions and secondary_choice in enhancement_functions[primary_choice]:
        func = enhancement_functions[primary_choice][secondary_choice]
        print(f'called Function: {func}')
        enhanced_img = func(img)
        return to_base64(enhanced_img)
    return to_base64(img)

def to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def histogram_equalization(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def sharpening(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)

def dehaze_image(image):
    return cv2.xphoto.createSimpleWB().balanceWhite(image)

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def retinex(image):
    retinex_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return retinex_image

def dncNN(image):
    dncNN_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return dncNN_image

def redNet(image):
    redNet_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return redNet_image

def vdsr(image):
    vdsr_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return vdsr_image

def edsr(image):
    edsr_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return edsr_image


def srcnn(image):
    srcnn_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return srcnn_image


def resnet_gan(image):
    resnet_gan_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return resnet_gan_image



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        primary_choice = request.form.get('primaryDropdown')
        if primary_choice=='Mathematical':
            secondary_choice = request.form.get('secondaryDropdownMath')
        elif primary_choice=='Traditional':
            secondary_choice = request.form.get('secondaryDropdownTraditional')
        elif primary_choice=='Novelty':
            secondary_choice = request.form.get('secondaryDropdownNovel')
        
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            enhanced_image = process_image(file_path, primary_choice, secondary_choice)
            original_image = to_base64(cv2.imread(file_path))
            
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
            print(secondary_choice)
            return render_template('result.html', 
                                 enhanced_image=enhanced_image, 
                                 original_image=original_image,
                                 secondary_choice =secondary_choice )
                                 
    return render_template('index.html')

@app.route('/about')
def about():    
    return render_template('aboutus.html')

@app.route('/contactus')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact_form', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Capture form data
        email = request.form.get('email')
        mobile = request.form.get('mobile')
        query = request.form.get('query')

        # Append the data to the CSV file
        with open('contact_submissions.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([email, mobile, query])  # Add row to CSV

        # Optionally, redirect or show success message
        return redirect(url_for('contact_success'))

@app.route('/contactus/success')
def contact_success():
    return render_template('contact_success.html')

if __name__ == '__main__':
    app.run(debug=False, port=3000)
