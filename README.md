# safe_driver
This project has two parts:
- A drowziness detection model (it detects via webcam if the driver's eyes are open closed or semi-closed)
- Two sensors, a dosimeter to measure alcohol in his breath and an accelerometer to predict accidents
****  
# Setup

**clone this repository (or download it):**

`git clone https://github.com/Mohamed-Dhouioui/projet_salah.git`

**Install required libraries via:**
- `pip install -r requirements.txt` on windows
- `pip3 install -r requirements.txt` on Raspberry pi

**Install dlib if you're working on Anaconda on windows via:**
- `conda install -c conda-forge dlib `

**On raspberry pi via:**
- `pip3 install dlib`

****
# Modules

- **drowsiness:** contains the webcam eyes detector, run via
`python detector_test.py`
- **images:** contains images for the sensors used in this project
- **sensors:** contains the modules for both sensors
