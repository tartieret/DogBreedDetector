from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
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

# generate urls for static files


# Check if input file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def homepage():

    # no description
    description = None

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', "error")
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file', "error")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            description = "This is a cat"

        else:
            flash('Invalid file type', "error")
            
    return render_template('index.html', description=description )


