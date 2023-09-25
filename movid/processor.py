import os
import glob
import datetime

from tqdm import tqdm  # for progress bars

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from movid.task import Task

class Processor:
    # this class represents a process in which potentially multiple movement video files will be processed.
    # It needs to be given a specified directory of videos to process.
    # Videos can be selectively included or excluded so that previously processed files are not repeated needlessly
    # or to focus only on certain participants or task types.
    # It will also need to specify a folder in which annotated videos can be saved and where numerical data is stored
    # (i.e. tables of landmark coordinates per frame).

    # most of the work is in the init function, which does all the config and set-up for the process:
    def __init__(self,
                 input_video_folder = 'videos',  # relative to the working directory
                 specific_videos = None,  # or a list of specific literal file names within input_video_folder
                 video_suffix = '.MOV',  # likely case-sensitive
                 task_types = ['fta', 'hoc'],  # specify at least one of the filename task codes (case-insensitive)
                 track = ['hands', 'face', 'pose'],  # specify at least one model (currently just 'hands' and/or 'face')
                 model_folder = 'models',  # MediaPipe model files location
                 output_video_folder = 'annotated_videos',
                 output_data_folder = 'landmark_data'):

        self.model_folder = model_folder
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
        # within that folder. In that case, no recursive search is done, and only the specified files (with sub-path
        # within the parent input_video_folder if necessary) will be processed.

        if specific_videos is None: # recursively identify all videos in the folder
            possible_videos = glob.glob(pathname = f'{self.input_video_folder}/**/*{video_suffix}', recursive = True)

            for path in possible_videos:
                pathname, filename = os.path.split(path)
                for task_type in task_types:
                    if task_type.lower() in filename.lower():
                        self.input_video_paths.append(path)
            print(f'### {len(possible_videos)} videos found. {len(self.input_video_paths)} selected by task.')
        else:  # only get the specified files
            for filename in specific_videos:
                path = f'{self.input_video_folder}/{filename}'
                self.input_video_paths.append(path)
            print(f'### {len(self.input_video_paths)} videos specified.')

        # create a string to use as a suffix for output files, to distinguish output when using different model
        # combinations:
        self.features = '-'.join(track)

        # create the mediapipe detectors needed for each feature to be tracked (hands, face, etc):
        self.detector_options = []
        if 'hands' in track:
            # set options:
            base_hand_options = python.BaseOptions(model_asset_path = f'{self.model_folder}/hand_landmarker.task')
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

        if 'face' in track:
            # see the face detection docs here:
            # https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python
            # set options:
            base_face_options = python.BaseOptions(model_asset_path = f'{self.model_folder}/face_landmarker.task')
            # note the RunningMode.VIDEO setting. This is needed so that information can carry over
            # from one frame to the next (we also need to provide timestamps). This produces much
            # higher quality results than analysing each frame in isolation, as if it was a still image:
            face_options = (
                vision.FaceLandmarkerOptions(base_options = base_face_options,
                                             running_mode = mp.tasks.vision.RunningMode.VIDEO))
            # these parameters aren't documented but need to be set to avoid exceptions:
            face_options.rotation_degrees = 0
            face_options.region_of_interest = None

            self.detector_options.append({'type': 'face', 'options': face_options})

    # once the configuration is done, can simply run the process. This is in a separate function so that
    # it is only invoked once the user has had a chance to see the output of the __init__ function,
    # which lists the number of videos to be processed. If that 'preflight' shows an incorrect number, it
    # gives the user a chance to try the config again.
    def run(self):

        start = datetime.datetime.now()
        print(f'Started processing at {str(start).split(".")[0]}.')  # remove the microseconds

        files_progress = tqdm(iterable = range(len(self.input_video_paths)),
                              desc = 'Videos: ',
                              unit = 'video',
                              leave = True)

        for i, video in enumerate(self.input_video_paths):
            task = Task(parent_proc = self, video_path = video)
            task.analyse_video()

            files_progress.write(' ' + video)
            files_progress.update()
            files_progress.refresh()

        end = datetime.datetime.now()
        print(f'Finished processing at {str(end).split(".")[0]}.')
        duration = end - start
        print(f'Time taken: {str(duration).split(".")[0]}.')
