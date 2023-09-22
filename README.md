# movid

`movid` (pronounced _moo vid_) is a Python package that uses the [MediaPipe](https://developers.google.com/mediapipe)
computer vision package to automatically track anatomical landmarks in videos of people with Parkinson's disease. The
goal is to easily extract quantitative measures of movement disorder symptoms and features.

## Installation

### From scratch

You can skip this stage if you are happy to install `movid` into your existing Python
installation. But if you would like to ring-fence it in its own clean environment, so 
that there are no package dependency conflicts, install Anaconda. Then from the terminal,
use it to manage environments, including one called, say, `analysis` with Python 
3.11 installed:

```commandline
conda create -n analysis python=3.11
conda activate analysis
```


### The `movid` package

From the terminal, this will install the `movid` package from the default branch 
(`main`) in its Gitgub repository into your local Python:

```commandline
pip install git+https://github.com/nzbri/movid.git
```

This will also install its two stated dependencies (`mediapipe` and `pandas`). Mediapipe 
will in turn install its own set of dependencies if needed (e.g. OpenCV).

To install some other branch from the repository, append the branch name:

```commandline
pip install git+https://github.com/nzbri/movid.git@some-branch
```

## Example usage

### Overview
Within a Python `.py` script or Jupyterlab notebook, there are three steps:

1. Import the `movid` package.
2. Create and configure a `movid.Processor` to analyse specified videos.
3. Run it.
4. Wait. Applying just the hand detector, an M2 MacBook Air takes approximately 12 times the duration of a video to 
   analyse it. This time will grow if multiple features are tracked. For example, processing both the hands and the face
   takes approximately 33 times the video duration on that laptop. An annotated video is exported, as well as a csv of
   landmark coordinates on each frame.

### Folder structure and organisation

- Put all the videos within one parent folder, which must be specified in the `input_video_folder` parameter.
  Videos can be flat or nested within sub-folders of that folder - the processor will search the parent folder
  recursively, so videos can be grouped within sub-folders for each participant, for example.
- The MediaPipe model files (e.g. `hand_landmarker.task`, `face_landmarker.task`) must be stored together within a
  specified folder (the `model_folder` parameter).
- Create and specify the locations of (initially empty) folders to store the annotated videos (the
  `output_video_folder` parameter) and the landmark data files (`output_data_folder` parameter).

Default names for all of those folders are provided in the `movid.Processor` constructor 
(`'videos'`, `'models'`, `'annotated_videos'`, `'landmark_data'`).

The landmark data CSV output file is compressed via gzip, making the files approximately 75% smaller. These files can be
read directly by the R `readr::read_csv` function without needing decompression first.

### `movid.Processor()` API

```python
processor = 
  movid.Processor(input_video_folder = 'videos',
                 specific_videos = None, # or a list of specific literal file names within input_video_folder
                 track = ['hands', 'face', 'pose'],  # specify at least one (currently just 'hands' and/or 'face')
                 task_types = ['fta', 'hoc'], # specify at least one of the task codes (case-insensitive)
                 model_folder = 'models', # MediaPipe model files location
                 output_video_folder = 'annotated_videos',
                 output_data_folder = 'landmark_data')
```

### Example script
This simple example will recursively find all videos in the default `'videos'` folder, select only those that are of the
finger tapping task, and apply the `hand_landmarker.task` MediaPipe model:

```python
import movid

processor = movid.Processor(track = ['hands'], task_types = ['fta'])
processor.run()

```

## TODO
- Add face feature number to each line of csv output (as for the hand feature names).
- Implement pose model (and potentially the "holistic" model).
- Handle paths properly rather than by concatenating strings.
- Sort the reversing of left and right handedness. Should be simple for hand tracking but may be a bigger issue for 
  pose tracking?
- Distinguish tasks that have (or don't have) a `_c` suffix.
- Including the list of features being tracked within each output file name doesn't seem to be working.
- Give better progress info to the user (such as within-video progress bars).
- Also log the start date and time of the processing run. Could then estimate the end time of the analysis.
