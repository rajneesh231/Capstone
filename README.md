# AquaVision: Enhancing Underwater Imagery

AquaVision is an innovative solution aimed at tackling common issues in underwater photography such as color distortion, low contrast, noise, and resolution loss. By leveraging advanced image processing techniques and deep learning models, AquaVision significantly enhances the quality of underwater images, making them suitable for applications in marine biology, underwater archaeology, industrial surveys, and more.

## Features
- **Image Enhancement Techniques**:
  - Traditional methods: Histogram Equalization, Image Denoising, and Color Correction.
  - Deep Learning: Models including VDSR, RedNet, DnCNN, EDSR, and SRCNN.
  - GAN-based Enhancement: ResNet GAN for high-quality restoration and noise reduction.

- **Applications**:
  - Marine exploration and conservation.
  - Underwater industrial inspections.
  - Search and rescue operations.

## Live Website
Try AquaVision live: [AquaVision Website](https://www.rajneeshb.live)

## Getting Started

### Prerequisites
- **Hardware**:
  - GPU-enabled system for model training or use of the pre-trained models.
  - Minimum: Intel i5 processor, 16 GB RAM, NVIDIA RTX 3060 (recommended).
- **Software**:
  - Python 3.9+
  - TensorFlow, PyTorch, OpenCV, Flask/Django for web deployment.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aquavision.git
   cd aquavision
   pip install -r requirements.txt
   python app.py
   ```
2. Docker Image (Alternative to Step 1):
   ```bash
   docker pull rajneesh768/flask-app
   docker run -p 3000:3000 rajneesh768/flask-app
   ```
3. Access The App:
   You can access the local deployment by typing below in your preferred browser
   ```bash
   localhost:3000
   ```
   Alternative, You may access the service on our deployed website at
   ```bash
   https://www.rajneeshb.live
   ```


### Authors
- Divyam Malik
- Rohan Gulati
- Abhinandan Sharma
- Rajneesh Bansal
- Anirudh Bansal
  
Under the mentorship of __Dr. Chinmaya Panigrahy__, Thapar Institute of Engineering and Technology.



