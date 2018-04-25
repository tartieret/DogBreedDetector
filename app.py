from flask import Flask, render_template, flash, request, redirect, url_for
from flask import jsonify
from werkzeug.utils import secure_filename
from classifier import get_breed, get_type      
import os


# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR,"upload")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# We need to generate a secrete key to allow sessions
# in production, set an ENV variable
app.secret_key = os.environ.get(
    'SECRET_KEY', 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT')  


        
# --------------------------------------------------
# Message display for the user
# --------------------------------------------------

def build_message(image_type, breed):
    message = "Unknown type"
    if image_type=="human":
        message = "Hello human! You look like a... {}!".format(breed)
    elif image_type=="dog":
        message = "Hello dog! You are a... {}!".format(breed)

    return message


# --------------------------------------------------
# Manage Errors
# --------------------------------------------------

class InvalidImage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidImage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

# --------------------------------------------------
# Manage requests
# --------------------------------------------------

# Check if input file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/analysis', methods=['POST'])
def analysis():

    # no description
    description = ""
    error = ""

    # check if the post request has the file part
    if 'file' not in request.files:
        raise InvalidImage(message = 'No file part')

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        raise InvalidImage(message = 'No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_path)
        file.close()

        try:
            
            # check if the image is a human or a dog
            image_type = get_type(full_path)
            print('Image type: ', image_type)

            # find the closest dog breed
            breed = get_breed(full_path)
            print('Breed: ', breed)

            # build the description message
            description = build_message(image_type, breed)

        except Exception as error:
            print(str(error))
            raise InvalidImage(message = 'This is neither a human nor a dog')
            
    return jsonify(description=description)


# Homepage
@app.route('/', methods=['GET'])
def homepage():           
    return render_template('index.html')


