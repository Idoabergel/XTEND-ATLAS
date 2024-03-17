# XTEND - ATLAS 
# Robot-Dog Collision Detection Project

## Overview
This project is designed as an educational assignment, introduce you to the basics of Artificial Intelligence (AI) and Computer Vision (CV) using Python.
The main objective is to analyze footage to detect people and determine if the robot is on a collision path with them.
This project utilizes a pre-trained YOLOv5 model for person detection.

## The Mission
- Keep people safe from robots on busy streets.
- Analyze robot videos to detect potential collisions.
- Get creative with alerts; sounds, flashing messages, anything goes!


## Prerequisites
- Python 3.8 or newer
- pip (Python package installer)

## Setup and Installation

### 1. Clone the Repository
First, clone this repository to your local machine using Git. Open your terminal and run:

```bash
git clone https://github.com/Idoabergel/XTEND-ATLAS.git
cd XTEND-ATLAS
```

##  Install Dependencies
This project requires certain Python packages to run. Install them using the following command:

```bash
pip install -r requirements.txt
```

## Download Pre-trained Model
The Python script will automatically download the pre-trained YOLOv5 model the first time you run it.
You can also manually download the model in this git folder. 

## Download The videos
Download the test videos using this link and copy them into the 'videos' folder.
```bash
https://xtend-content.wetransfer.com/downloads/5899119aa082924ca6558dd11d567dae20240314165344/85075a193eb4b519535c9b2b61dc8c3020240314165344/77a906
```

## Running the Code
To start the person detection and collision prediction, navigate to the project directory in your terminal and run:

```bash
python detect_people.py
```

Before running the script, ensure you modify the video_path variable in detect_people.py to point to the path of your robot footage video file.

## How It Works
The script uses the YOLOv5 model to detect people in each frame of the provided video. It draws green bounding boxes around detected people. The logic for determining if the robot is about to collide with any of the detected people needs to be implemented as part of the assignment.


## Contributing

Let Your Ideas Fly!
Got an idea? Share it! We're all ears for your genius. Just hit us up with your thoughts or fixes, and let's whip up some magic together!

## Additional Notes

- XTEND and ATLAS Collaboration: Where high schoolers become the last line of defense against the robot apocalypse.
- Project Insight: Proof that the journey to save the world begins with the most heroic task of all: homework.