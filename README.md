# Music rearranger
Rearrange a music recording so that it matches a new desired duration. Code for "Music rearrangement using hierarchical segmentation" ICASSP 2023 paper.

### Disclaimer
This code is not 100% reflecting the methods described in the paper. Most notably, the path finding approach has been replaced with a simpler one until I manage to debug the original. This simpler one, however, might sometimes fail to find a path, and does not prioritize transition points based on their similarity. 

I'm aiming to develop this package further, including doing some work on finding more optimal default parameters for the segmentation and transition point identification configuration. Submiting issues and pull request is welcomed and appreciated.

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

### Running

There are two key aspects to this rerrangement method: 1. segmentation and 2. path finding using the identified transitions points. You can compute the segmentation information once, and use it to rearrange the piece multiple times. So the first command you can run is:
```bash
python rearrange.py --input_audio /path/to/audio/file --target_time 60
```
where `--target_time` is the desired duration of the rearrangement in seconds.

After having computed the rearrangement, a pickle file and an audio file will be created. You can use the pickle file as an argument so that you don't recompute the structure every time.
```bash
python rearrange.py --input_audio /path/to/audio/file --input_seg /path/to/segmentation/pickle/file --target_time 60
```
Other useful options include:
`--seg_method`: segmentation method to use. This currently includes the Salamon et al. 2021 segmentation method (`precise`) which is more accurate but slower, as well as the McFee & Ellis 2014 segmentation method (`fast`) which is less accurate but faster.
`--use_gpu` (flag): whether to use the GPU for the feature computation for the `precise` segmentation.
`--output_dir`: output directory for audio and pickle file.
`--config`: path to configuration file with various segmentation, transition identification, and path finding parameters.
`--plot` (flag): whether to save plots of the features and segmentation that is computed.

### Reference
```
@inproceedings{music-rearranger,
	Author = {C. Plachouras and M. Miron},
	Booktitle = {ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
	Month = {Jun.},
	Title = {Music rearrangement using hierarchical segmentation},
	Year = {2023}}
```