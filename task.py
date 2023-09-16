import cv2
import os
import time
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class Task:
    # This class represents a recording of a single UPDRS task. It needs to have a path to a source video to be
    # analysed. This will then be processed using one or more MediaPipe detectors (e.g. for detecting hand, face, or
    # pose landmarks).
    # It can then produce a video which is annotated (i.e. overlaid with the detected landmarks on each frame).
    # It will also produce a table of numeric coordinates (generally xyz, absolute or normalised) for each landmark on
    # each frame.

    def __init__(self, parent_proc, video_path):

        self.video_in = cv2.VideoCapture(video_path)
        if not self.video_in.isOpened():
            print(f'Error opening the video file {video_path}')
            return

        self.fps = round(self.video_in.get(cv2.CAP_PROP_FPS), 3)
        self.width = int(self.video_in.get(3))
        self.height = int(self.video_in.get(4))

        self.video_in_folder_path, self.video_in_filename = os.path.split(video_path)
        filename_parts = self.video_in_filename.split('_')
        if len(filename_parts) >= 3:
            self.date = filename_parts[0]
            self.subject = filename_parts[1].upper()
            self.task = filename_parts[2].upper()
        else:
            self.subject = self.video_in_filename
            self.date = 'not parsed'
            self.task = 'not parsed'

        self.video_out = None  # not initialised until process_video() called
        # the name of a subfolder where the annotated video will be saved (should be different to the folder containing
        # the original source videos, to avoid over-writing source data):
        self.video_out_folder_path = parent_proc.output_video_folder
        self.video_out_filename = self.video_in_filename[:-4] + '_labelled.mp4'

        # this 4-byte code controls the video codec to be used. See
        # https://gist.github.com/takuma7/44f9ecb028ff00e2132e for Mac-compatible values.
        # avc1 compresses well but seemed to produce keyframe artefacts:
        # fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        # mp4v compresses less well but gives quality comparable to original colour file:
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        # from the parent processor, work through its list of detector options (e.g. one each for hands, face, pose) and
        # from each instantiate eg a HandLandmarker object, which needs to be done afresh for each video:
        self.detectors = []
        for item in parent_proc.detector_options:

            if item['type'] == 'hands':
                detector = vision.HandLandmarker.create_from_options(item['options'])
                self.hand_landmark_names = [mark.name for mark in solutions.hands.HandLandmark]

            if item['type'] == 'face':
                detector = vision.FaceLandmarker.create_from_options(item['options'])

            self.detectors.append({'type': item['type'],
                                   'detector': detector,
                                   'options': item['options']})

        # initialise an empty  dataframe to hold the coordinates of the detected landmarks:
        self.output_data = pd.DataFrame()

    def analyse_video(self):

        self.video_out = (
            cv2.VideoWriter(filename = f'{self.video_out_folder_path}/{self.video_out_filename}', fourcc = self.fourcc,
                            fps = self.fps, frameSize = (self.width, self.height), isColor = True))

        print(f'Processing {self.video_in_filename}')
        start_time = time.time()

        while self.video_in.isOpened():
            # capture frame-by-frame
            success, bgr_image = self.video_in.read()

            if success:
                time_stamp = int(self.video_in.get(cv2.CAP_PROP_POS_MSEC))  # time in ms
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_image)

                annotated_image = None

                for detector in self.detectors:
                    #  detect landmarks from the input image:
                    detection_result = detector['detector'].detect_for_video(image = mp_image,
                                                                             timestamp_ms = time_stamp,
                                                                             image_processing_options = detector['options'])

                    # extract the coordinates:
                    coords = self.get_coords(detection_result, detector['type'])
                    coords['time_stamp'] = time_stamp
                    self.output_data = pd.concat([self.output_data, coords], ignore_index = True)

                    # draw the coordinates:
                    if annotated_image is None:
                        annotated_image = bgr_image
                    annotated_image = self.draw_landmarks_on_image(rgb_image = annotated_image,
                                                                   detection_result = detection_result,
                                                                   detector_type = detector['type'])
                self.video_out.write(annotated_image)
            else:
                break

        # tidy up:
        self.video_in.release()
        self.video_out.release()

        self.output_data['task'] = self.task
        self.output_data['date'] = self.date
        self.output_data['subject'] = self.subject
        self.output_data.to_csv(f'{self.video_out_folder_path}/{self.video_in_filename[:-4]}.csv')

        print(f'  Duration: {round(time_stamp / 1000, 1)}')
        print(f'  Time taken: {round(time.time() - start_time, 1)} s')
        print(f'  Saved as: {self.video_out_filename}')

    def get_coords(self, detection_result, detector_type):
        # this function is passed:
        #  detection_result: the output from the function
        #                    mediapipe.tasks.python.vision.HandLandmarker.detect_for_video()
        #                    previously applied to that frame. This object contains the
        #                    image (and world) coordinates of the various landmarks.
        # this functions returns a dataframe of coordinates for each landmark.

        output = pd.DataFrame()

        if detector_type == 'hands':
            features = detection_result.hand_world_landmarks
        elif detector_type == 'face':
            features = detection_result.face_landmarks

        for i, landmarks in enumerate(features):

            coords = [{'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                      for landmark in landmarks]

            temp_df = pd.DataFrame.from_records(coords)
            temp_df['detector_type'] = detector_type
            if detector_type == 'hands':
                temp_df['landmark'] = self.hand_landmark_names  # assumed to be in same order from 0 to 20
                temp_df['side'] = detection_result.handedness[i][0].display_name
            elif detector_type == 'face':
                temp_df['landmark'] = 'NA (face)'
                temp_df['side'] = 'NA (face)'

            output = pd.concat([output, temp_df], ignore_index = True)

        return output

    def draw_landmarks_on_image(self, rgb_image, detection_result, detector_type):

        # this function is passed:
        #  rgb_image: a single frame from a video, and
        #  detection_result: the output from the function
        #                    mediapipe.tasks.python.vision.HandLandmarker.detect_for_video()
        #                    previously applied to that frame. This object contains the
        #                    image (and world) coordinates of the various landmarks.

        margin = 10  # pixels
        font_size = 1
        font_thickness = 1
        handedness_text_colour = (88, 205, 54)  # vibrant green

        annotated_image = np.copy(rgb_image)

        # TODO - generalise to other detectors:
        if detector_type == 'hands':
            hand_landmarks_list = detection_result.hand_landmarks
            handedness_list = detection_result.handedness

            # Loop through the detected hands to visualize.
            for (hand_landmarks, handedness) in zip(hand_landmarks_list, handedness_list):
                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

                hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x = landmark.x,
                                                                                      y = landmark.y,
                                                                                      z = landmark.z)
                                                      for landmark in hand_landmarks])

                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = annotated_image.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - margin

                # draw handedness (left or right hand) on the image.
                # this will currently be incorrect, as mediapipe assumes the camera is front-facing:
                cv2.putText(annotated_image, f'{handedness[0].category_name}',
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            font_size, handedness_text_colour, font_thickness, cv2.LINE_AA)

        if detector_type == 'face':
            # TODO see https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb

            for face_landmarks in detection_result.face_landmarks:
                # Draw the face landmarks.
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

                face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x = landmark.x,
                                                                                      y = landmark.y,
                                                                                      z = landmark.z)
                                                      for landmark in face_landmarks])

                solutions.drawing_utils.draw_landmarks(
                    image = annotated_image,
                    landmark_list = solutions.connection_drawing_spec.face_landmarks_proto,
                    connections = solutions.face_mesh_connections.FACEMESH_TESSELATION,
                    connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_contours_style())

        return annotated_image
