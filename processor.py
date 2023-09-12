import os
import glob

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from task import Task

class Processor:
    # this class represents a process in which potentially multiple movement video files will be processed.
    # It will need to have a specified directory of videos to process.
    # Videos can be selectively included or excluded so that previously processed files are not repeated needlessly
    # or to focus only on certain participants or task types.
    # It will also need to specify a folder in which annotated videos can be saved and where numerical data is stored
    # (i.e. tables of landmark coordinates per frame).

    def __init__(self,
                 detect = ['hands', 'face', 'pose'],  # specify at least one (currently just 'hands')
                 input_video_folder  = 'videos',
                 output_video_folder = 'mp_processed_videos',
                 output_data_folder = 'mp_landmark_data'):

        self.input_video_paths = []
        self.input_video_folder = input_video_folder
        self.output_video_folder = output_video_folder
        self.output_data_folder = output_data_folder

        self.detector_options = []

        if 'hands' in detect:
            # set options:
            base_hand_options = python.BaseOptions(model_asset_path = 'models/hand_landmarker.task')
            # note the RunningMode.VIDEO setting. This is needed so that information can carry over
            # from one frame to the next (we also need to provide timestamps). This produces much
            # higher quality results than analysing each frame in isolation, as if it was a still image:
            hand_options = (
                vision.HandLandmarkerOptions(base_options = base_hand_options,
                                             running_mode = mp.tasks.vision.RunningMode.VIDEO,
                                             num_hands = 2))
            # these parameters aren't documented but need to be set to avoid exceptions:
            hand_options.rotation_degrees = 0
            hand_options.region_of_interest = None

            self.detector_options.append({'type': 'hands', 'options': hand_options})

    def process_videos(self, videos, task_types = ['fta']):
        # input_videos can be a list of literal filenames, in which case they will each get prefixed with the path
        # to the input video folder.
        # if it is a single string literal, that is assumed to be the path to a folder and we will iterate recursively
        # over that folder, collecting all video file names that contain one of the task type strings.

        if type(videos) is str:
            for path in glob.glob(videos, recursive = True):
                pathname, filename = os.path.split(path)
                for task_type in task_types:
                    if task_type in filename:
                        self.input_video_paths.append(path)
        elif type(videos) is list:
            for filename in videos:
                path = f'{self.input_video_folder}/{filename}'
                self.input_video_paths.append(path)

        for video in self.input_video_paths:
            task = Task(parent_proc = self, video_path = video)
            task.analyse_video()