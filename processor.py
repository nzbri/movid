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

    # most of the work is in the init function, which dos all the config and set-up for the process:
    def __init__(self,
                 input_video_folder = 'videos',
                 specific_videos = None,, # or a list of specific literal file names within input_video_folder
                 track = ['hands', 'face', 'pose'],  # specify at least one (currently just 'hands')
                 task_types = ['fta', 'hoc'], # specify at least one
                 output_video_folder = 'annotated_videos',
                 output_data_folder = 'landmark_data'):

        self.input_video_folder = input_video_folder
        self.input_video_paths = [] # will get populated with the actual video filenames
        self.output_video_folder = output_video_folder
        self.output_data_folder = output_data_folder

        # we must always provide a folder in which the source videos can be found.
        # if specific_videos is None, then that folder will be searched recursively (i.e. including
        # within any of its sub-folders) to identify all files (all assumed to be valid videos). Only files
        # that include one of the task types (e.g. 'fta' for finger tapping) will be included in the list
        # to be processed.
        # if specific_videos is not None, then it should be a list containing at least one video filename
        # within that folder. In that case, no recursive search is done, and only the specified files will
        # be processed. They are assumed to be within input_video_folder, so no folder path is required.

        if specific_videos is None: # recursively identify all videos in the folder
            possible_videos = glob.glob(self.input_video_folder, recursive = True)

            for path in possible_videos:
                pathname, filename = os.path.split(path)
                for task_type in task_types:
                    if task_type.lower() in filename.lower():
                        self.input_video_paths.append(path)
            print(f'### {len(possible_videos)} videos found. {len(self.input_video_paths)} selected by task.')
        else: # only get the specified files
            for filename in specific_videos:
                path = f'{self.input_video_folder}/{filename}'
                self.input_video_paths.append(path)
            print(f'### {len(self.input_video_paths)} videos specified.')

        # create the mediapipe detectors needed for each feature to be tracked (hands, face, etc):
        self.detector_options = []
        if 'hands' in track:
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

        # TODO:
        if 'face' in track:
            # see the face detection docs here:
            # https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python
            pass

    # once the configuration is done, can simply run the process. This is in a separate function so that
    # it is only invoked once the user has had a chance to see the output of the __init__ function,
    # which lists the number of videos to be processed. If that 'preflight' shows an incorrect number, it
    # gives the user a chance to try the config again.
    def run(self):
        for video in self.input_video_paths:
            task = Task(parent_proc = self, video_path = video)
            task.analyse_video()