# Instructions

- Add the song you want to rearrange to the `audio/` directory.
- Run `make run IN=<name of audio file>`
- By default it tries to make a 30 second rearrangmenet, add `DUR=<desired seconds` to the command above to change the target time

## Dockerfile background

Plachouras code requires Python 3.7 and Tensorflow 2.7. The appropriate NVIDIA Tensorflow container, 22.01, can be found [here](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_22-02.html#rel_22-02). It comes with Python 3.8 so Python 3.7 must be installed manually.
