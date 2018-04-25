from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image  
from keras.models import model_from_json
from extract_bottleneck_features import *   
import numpy as np
from tqdm import tqdm
from breeds import BREEDS
import cv2
import os


# --------------------------------------------------
# Dog detector
# -------------------------------------------------

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# --------------------------------------------------
# Human detector
# -------------------------------------------------

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
eye_glasses_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        # try to detect eyes
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # if no eyes, detected, look for glasses
        if len(eyes)==0:
            eyes_glasses = eye_glasses_cascade.detectMultiScale(roi_gray)
            
    return (len(faces) > 0) and (len(eyes)>=1 or len(eyes_glasses)>=1)

# --------------------------------------------------
# Dog Breed Classifier
# -------------------------------------------------

# load inception model
def load_Inception_model():
    # load json model
    json_file = open(os.path.join('saved_models','Inception_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(os.path.join('saved_models','weights.best.Inception.hdf5'))
    return loaded_model

Inception_model= load_Inception_model()

def Inception_predict_breed(img_path):
    """Returns the dog breed predicted by Inception"""
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Inception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return BREEDS[np.argmax(predicted_vector)]

# --------------------------------------------------
# Dog Breed Classifier
# -------------------------------------------------

# check if the image is human or dog
def get_type(img_path):
    print("Working with: ", img_path)

    # first, test if this is a human
    if face_detector(img_path):
        return "human"
    elif dog_detector(img_path):
        return "dog"
    else:
        raise ValueError('Neither a dog nor a human')

def get_breed(img_path):
    # predict the breed
    breed = Inception_predict_breed(img_path)       
    return breed



