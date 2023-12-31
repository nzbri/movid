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
3.11 installed in it:

```commandline
conda create -n analysis python=3.11
conda activate analysis
```


### The `movid` package

From the terminal, this will install the `movid` package from the default branch 
(`main`) in its GitHub repository into your local Python:

```commandline
pip install git+https://github.com/nzbri/movid.git
```

This will also install its three stated dependencies (`mediapipe`, `pandas`, and `tqdm`). Mediapipe 
will in turn install its own extensive set of dependencies if needed (e.g. OpenCV).

To install some other branch or tag from the repository, append its name:

```commandline
pip install git+https://github.com/nzbri/movid.git@some-branch
```

## Example usage

### Overview
Within a Python `.py` script or Jupyterlab notebook, there are three steps:

1. Import the `movid` package.
2. Create and configure a `movid.Processor` to analyse specified videos.
3. Run it.
4. Wait. Applying just the hand detector, an M2 MacBook Air processes approximately 18 frames per second, which for a
   240 Hz recording will take 13 times the duration of the video to complete. This time will grow if multiple features
   are tracked. For example, processing hands, face, and pose runs at approximately 3–5 frames per second on that 
   laptop. This performance is, however, achieved using only a single core. Substantial improvements might be possible
   through implementing parallel computing on multiple cores.

   An annotated video is exported along with a hopefully representative .JPG thumbnail (taken at half-way through the 
   video). Quantitative data is exported as a zipped CSV file of landmark coordinates on each frame.

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
  movid.Processor(input_video_folder = 'videos',  # relative to the working directory
                 specific_videos = None, # or a list of specific literal file names within input_video_folder
                 video_suffix = '.MOV',  # likely case-sensitive
                 task_types = ['fta', 'hoc'],  # if not listing specific videos, give at least one task code to be 
                                               # searched for in filenames (case-insensitive)
                 track = ['hands', 'face', 'pose', 'holistic'],  # specify at least one model (holistic not implemented)
                 model_folder = 'models',  # MediaPipe model files location
                 output_video_folder = 'annotated_videos',
                 output_data_folder = 'landmark_data')
```

### Example script
This simple example will recursively find all videos in the default `'videos'` folder, select only those that are of the
finger tapping task, and apply the `hand_landmarker.task` MediaPipe model:

```python
import movid

processor = movid.Processor(task_types = ['fta'], track = ['hands'])
processor.run()

```

## TODO
- Implement the "holistic" model. Might not be feasible - this seems to be waiting on an upgrade path that has already
  been done for the other models in the MediaPipe project
- Handle paths properly rather than by concatenating strings.
- Sort the reversing of left and right handedness. Should be simple for hand tracking but may be a bigger issue for 
  pose tracking?
- Distinguish tasks that have (or don't have) a `_c` suffix.
- Increase performance by implementing parallel processing – currently the code is constrained by running on only one 
  core.
- Use PCA to get primary axis of finger tapping or other tasks, regardless of hand orientation. Might be best as a
  post-processing step in R.
