# Music rearranger
Code for "Music rearrangement using hierarchical segmentation" paper.

*Note*: The code in this resository is preliminary, reflecting earlier experiments in this project, so it doesn't 100% match the methodologies described in the paper. The updated, stable code will be released soon.

### Installation

Install non-python dependencies:

* ffmpeg
* sox (with support for mp3)
* libsndfile

On Debian/Ubuntu you can install them with the following:
```
apt-get update --fix-missing && apt-get install libsndfile1 ffmpeg libsox-fmt-all sox -y
```

Then, create a python environment with python=3.7 and install the dependencies in `requirements.txt`.