################# Loading libraries and frameworks #################
from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
from datetime import datetime
from PIL import ImageGrab
from Controller.cnn import detect_face

#####################################################################

app = Flask(__name__)

################################## Haar Detector path ##################################
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
#########################################################################################

############### Haar Classifier creation ###############
face_class = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
########################################################

n_faces = 0

#################### Get video ####################
## Video path. By default, we get the webcam ## 
video_path=0
###################################################
camera = cv2.VideoCapture(video_path)
###################################################

# Function which allows us to detect the face #      

        
             #######################################################################################
############################################

######################## Routing to the face detection function ########################
@app.route('/video_feed')
def video_feed():
    
    return Response(detect_face.face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################################################
################# Main page #################




@app.route('/')
def index():
    return render_template('index.html')
#############################################
if __name__ == '__main__':
    app.run(debug=False)
