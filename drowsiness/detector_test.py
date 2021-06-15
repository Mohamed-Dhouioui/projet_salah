from detector import *

config = {
            "face_landmarks_model_path" : "models/haarcascade_frontalface_default.xml",
            "eye_model_path" : "models/weights.149-0.01.hdf5" ,
            "haarcascade_path" : "models/68_face_landmarks_predictor.dat"
        }
D = Detector(config)

