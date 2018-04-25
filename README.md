[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./app_screenshots/screenshot1.png "AppScreenshot1"
[image5]: ./app_screenshots/screenshot2.png "AppScreenshot2"

## Project Overview

### Deep Learning

In this project, I built a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

In the first part of the project, I worked in a Jupyter notebook to perform the following steps:
1. Use Haar feature-based cascade classifiers to detect human faces in images
2. Use a pre-trained (on ImageNet) ResNet-50 model to detect dogs in images
3. Design a CNN architecture to identify dog breeds
4. Use Transfer Learning from VGG16 to identify dog breeds
5. Use Transfer Learning from GoogLeNet to identify dog breeds

My own CNN architecture (step 3) reached a 35.76% accuracy on the test set, well above the minimum requirements for the project (1%). It was trained for 4 hours on a GPU. However, using transfer learning from the Inception/GoogLeNet was very successful with a final accuracy of 80.5%

Check the Jupyter notebook ```dog_app.ipynb``` for more details.

### Web application

In a second step, I built a Flask web application to serve the model through a Bootstrap/JQuery web interface. Here is the final result:

![Web application][image4]
![Web application][image5]

## Setup

### General

1. Clone the repository and navigate to the downloaded folder.
``` 
git clone https://github.com/tartieret/DogBreedDetector
cd DogBreedDetector
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`. If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset. Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

4. Donwload the [Inception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) for the dog dataset. Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system. If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

    - __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`):
    ```
    conda env create -f requirements/dog-linux.yml
    source activate dog-project
    ```
    - __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`):
    ```
    conda env create -f requirements/dog-mac.yml
    source activate dog-project
    ```
    **NOTE:** Some Mac users may need to install a different version of OpenCV
    ```
    conda install --channel https://conda.anaconda.org/menpo opencv3
    ```
    - __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):
    ```
    conda env create -f requirements/dog-windows.yml
    activate dog-project
    ```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

    - __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
    ```
    conda create --name dog-project python=3.5
    source activate dog-project
    pip install -r requirements/requirements.txt
    ```
    **NOTE:** Some Mac users may need to install a different version of OpenCV
    ```
    conda install --channel https://conda.anaconda.org/menpo opencv3
    ```
    - __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
    ```
    conda create --name dog-project python=3.5
    activate dog-project
    pip install -r requirements/requirements.txt
    ```
    
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
    
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
    - __Linux__ or __Mac__:
        ```
        KERAS_BACKEND=tensorflow python -c "from keras import backend"
        ```
    - __Windows__:
        ```
        set KERAS_BACKEND=tensorflow
        python -c "from keras import backend"
        ```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment.
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

### Web application

12. Define an environment variable:
```
export FLASK_APP=app.py
```
If you are on Windows, you will need to use ```set``` instead of ``export``

13. Run the Flask server
```
flask run
```
14. Open your browser and visit http://127.0.0.1:5000
