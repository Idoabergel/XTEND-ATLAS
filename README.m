# Robot-Dog Collision Detection Project

## Overview
This project is designed as an educational assignment, introduce them to the basics of Artificial Intelligence (AI) and Computer Vision (CV) using Python.
The main objective is to analyze footage to detect people and determine if the robot is on a collision path with them.
This project utilizes a pre-trained YOLOv5 model for person detection.

## Prerequisites
- Python 3.8 or newer
- pip (Python package installer)

## Setup and Installation

### 1. Clone the Repository
First, clone this repository to your local machine using Git. Open your terminal and run:

```bash
git clone <repository-url>          # REPLACE WITH GIT PATH
cd drone-collision-detection

##  Install Dependencies
This project requires certain Python packages to run. Install them using the following command:

```bash
pip install -r requirements.txt

## Download Pre-trained Model
The Python script will automatically download the pre-trained YOLOv5 model the first time you run it.

Running the Code
To start the person detection and collision prediction, navigate to the project directory in your terminal and run:

```bash
python detect_people.py

Before running the script, ensure you modify the video_path variable in detect_people.py to point to the path of your drone footage video file.

How It Works
The script uses the YOLOv5 model to detect people in each frame of the provided video. It draws green bounding boxes around detected people. The logic for determining if the drone is about to collide with any of the detected people needs to be implemented as part of the assignment.

Contributing
This project is designed for educational purposes and is open to contributions. If you have suggestions for improving the assignment or encounter any issues, please open an issue or a pull request.

License
This project is open-sourced under the MIT license. See the LICENSE file for details.

### Additional Notes:
Make sure to:
- Replace `<repository-url>` with the actual URL of your GitHub repository.
- Check the link to Ultralytics' GitHub repository to ensure it's current and accurate.
- Consider adding a `LICENSE` file to your repository if you reference it in the README.
- Adjust any paths, filenames, or specific details to match your project setup.


