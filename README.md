# Mask Detection and Facial Recognition
This project combines real-time mask detection with facial recognition to identify individuals not wearing masks in a controlled environment. It was developed as part of a TIPE (Travail d'Initiative Personnelle Encadrée) project.

## Project Structure
```bach
|-Detection
  |-mask_detection-main
    |-dataset
      |-with_mask
      |-without_mask
    |-face_detector
      |-deploy.prototxt
      |-mask_detector.model
      |-res10_300x300_ssd_iter_140000.caffemodel
    |-main.py
  |-myFaceReco
    |-eigenface
      |-eigenface0.png
          .
          .
      |-eigenface35.png
    |-gallery
    |-faceReco.py
    |-faceReco.ui
    |-faceRecoGUI.py
    |-faceRecoGUI.pyc
  |-MCOT_13220_49566.pdf
  |-Présentation TIPE.pdf
  |-detection.py
  |-live_detection.py
|_README
```

## Functionality
The project is divided into two main parts:

#### 1. Mask Detection:

- A Convolutional Neural Network (CNN) model, trained on a dataset of images with and without masks, is used for real-time detection of whether a person is wearing a mask.

- If a face without a mask is detected, the program extracts the face region and saves it for further identification.

#### 2. Facial Recognition:

- The extracted face is then compared to a database of known faces using PCA (Principal Component Analysis) and the eigenfaces method.

- The program calculates a similarity score between the extracted face and the faces in the database.

- If the similarity score exceeds a certain threshold, the name of the corresponding person is displayed.

## Technologies Used
- Python 3

- TensorFlow/Keras

- OpenCV

- NumPy

- SciPy

- PyQt4 (for the optional graphical interface)

## Installation
#### 1. Clone the repository:
```bach
git clone https://github.com/your_username/your_project_name.git
cd your_project_name
```
#### 2. Create a virtual environment (recommended):
```bach
python3 -m venv env
source env/bin/activate
```
#### 3. Install dependencies:
```bach
pip install -r requirements.txt
```
## Execution
#### Real-time Mask Detection:
```bach
python live_detection.py
```
#### Mask Detection on a Single Image:
```bach
python detection.py -i <image_path>
```
Replace `<image_path>` with the path to your image.

#### Facial Recognition (with graphical interface):
```bach
python faceReco.py
```
## Face Database
Place the reference face images in the `myFaceReco/gallery` folder.

Ensure that the images are in `PNG` format.

## Notes
- The similarity threshold for facial recognition can be adjusted in the code.

- The performance of facial recognition depends on the quality and size of the face database.

- The mask detection model may require fine-tuning depending on your dataset and specific needs.
