import csv
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from tensorflow.keras.models import load_model
from PIL import Image
import joblib
# Load the model
modellllll = joblib.load("models/resnetGAN.pkl")
paramssssss = modellllll['parameters']

# srcnn_model = joblib.load("models/srcnn_model.pkl")
srcnn_model = load_model('models/srcnn_model.h5', compile=False)
srcnn_model.compile(optimizer='adam') 
# Load the saved Keras model
edsr_model = load_model('models/edsr_best_model.h5', compile=False)
edsr_model.compile(optimizer='adam')  # Add your specific optimizer settings here

vdsr_Model = load_model("models/vdsr_best_model.h5", compile=False)
vdsr_Model.compile(optimizer='adam') 

dncnn_Model = load_model("models/lite_dncnn_best_model.h5", compile=False)
dncnn_Model.compile(optimizer='adam') 

rednet_Model = load_model("models/rednet_best_model.h5", compile=False)
rednet_Model.compile(optimizer='adam') 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
import logging

if not app.debug:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Only log errors
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img
def resizeImg(img, target_size):  # Ensure it's RGB
    img = img.resize(target_size)  
    return img



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, primary_choice, secondary_choice):
    enhancement_functions = {
        'Mathematical': {
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
    img = load_image(image_path)
    if primary_choice in enhancement_functions and secondary_choice in enhancement_functions[primary_choice]:
        func = enhancement_functions[primary_choice][secondary_choice]
        enhanced_img = func(img)
        return to_base64(enhanced_img)
    return to_base64(img)

def to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def histogram_equalization(image):
    image = np.array(image)
    # Ensure image is in uint8 format
    image = image.astype(np.uint8)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def sharpening(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1],
                       [0, -1, 0]])
    image = np.array(image)
    # Ensure image is in uint8 format
    image = image.astype(np.uint8)
    return cv2.filter2D(image, -1, kernel)


def dehaze_image(image):
    image = np.array(image)
    # Ensure image is in uint8 format
    image = image.astype(np.uint8)
    return cv2.xphoto.createSimpleWB().balanceWhite(image)

def gamma_correction(image, gamma=1.5):
    image = np.array(image)
    # Ensure image is in uint8 format
    image = image.astype(np.uint8)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def retinex(image):
    image = np.array(image)
    # Ensure image is in uint8 format
    image = image.astype(np.uint8)
    retinex_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return retinex_image

def dncNN(image):
    image = resizeImg(image, (128, 128))
    img_array = np.array(image)
    
    # Normalize the image
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Get prediction
    dncNN_image = dncnn_Model.predict(img_tensor)
    
    # Post-process the output
    output = np.squeeze(dncNN_image)  # Remove batch dimension
    output = np.clip(output * 255, 0, 255)  # Scale back to 0-255 range
    output = output.astype(np.uint8)  # Convert to uint8
    output = cv2.resize(output, (256, 256))
    return output

    
def ResnetGan(image,clip_limit,tile_grid):
    image = np.array(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    out1 = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out2 = out1.apply(l)
    out3 = cv2.merge((out2, a, b))
    enhanced_img = cv2.cvtColor(out3, cv2.COLOR_LAB2RGB)
    return enhanced_img

def redNet(image):
    image = resizeImg(image, (128, 128))
    img_array = np.array(image)
    
    # Normalize the image
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)
    redNet_image = rednet_Model.predict(img_tensor)
    output = np.squeeze(redNet_image)  # Remove batch dimension
    output = np.clip(output * 255, 0, 255)  # Scale back to 0-255 range
    output = output.astype(np.uint8)  # Convert to uint8
    output = cv2.resize(output, (256, 256))
    return output

def vdsr(image):
    image = resizeImg(image,(128,128))
    image = resizeImg(image,(128,128))
    img_array = np.array(image)
    
    # Normalize the image
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension and channel dimension if needed
    img_tensor = np.expand_dims(img_array, axis=0)
    vdsr_image = vdsr_Model.predict(img_tensor)
    output = np.squeeze(vdsr_image)  # Remove batch dimension
    output = np.clip(output * 255, 0, 255)  # Scale back to 0-255 range
    output = output.astype(np.uint8)  # Convert to uint8
    output = cv2.resize(output, (256, 256))
    return output

def edsr(image):
    image = resizeImg(image,(128,128))
    image = resizeImg(image,(128,128))
    img_array = np.array(image)
    
    # Normalize the image
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension and channel dimension if needed
    img_tensor = np.expand_dims(img_array, axis=0)
    edsr_image = edsr_model.predict(img_tensor)
    output = np.squeeze(edsr_image)  # Remove batch dimension
    output = np.clip(output * 255, 0, 255)  # Scale back to 0-255 range
    output = output.astype(np.uint8)  # Convert to uint8
    output = cv2.resize(output, (256, 256))
    return output



def srcnn(image):
    image = resizeImg(image,(128,128))
    image = resizeImg(image,(128,128))
    img_array = np.array(image)
    
    # Normalize the image
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension and channel dimension if needed
    img_tensor = np.expand_dims(img_array, axis=0)
    srcnn_image = srcnn_model.predict(img_tensor)
    output = np.squeeze(srcnn_image)  # Remove batch dimension
    output = np.clip(output * 255, 0, 255)  # Scale back to 0-255 range
    output = output.astype(np.uint8)  # Convert to uint8
    output = cv2.resize(output, (256, 256))
    return output



def resnet_gan(image):
    image = resizeImg(image,(256,256))
    resnet_gan_image = ResnetGan(image,clip_limit = paramssssss['clip_limit'],tile_grid = tuple(paramssssss['tile_grid_size']))
    return resnet_gan_image

def resizeImgorig(img, target_size):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


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
            if primary_choice!='Mathematical':
                original_image = resizeImgorig(cv2.imread(file_path),(256,256))
                original_image = to_base64(original_image)
            else:
                original_image = to_base64(cv2.imread(file_path))
                

            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
