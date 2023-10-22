import streamlit as st
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
from typing import Tuple
import pandas as pd

from beep import detect_beeps
from motion import detect_motion

#@st.cache_data
def convert_df(df):
    return df.to_csv(index=True).encode('utf-8')


################################################
# Object detection initialization and functions
################################################
# Download a trained Yolo model from HuggingFace hub
model_path = hf_hub_download(repo_id='arieg/spike-prime-robot-detection', filename='spike-prime-robot-detection.pt')
yolo = YOLO(model_path)

def robot_position( image ) -> Tuple[float, float]:
    # pick the first result
    result = yolo(image, conf=obj_detection_conf, iou=0.7)[0]

    if len(result.boxes.cls) == 0:  # no robot detected, zero bounding boxes
        return None
    else:
        return (float(0.5*(result.boxes.xyxy[0][0] + result.boxes.xyxy[0][2])),
                float(0.5*(result.boxes.xyxy[0][1] + result.boxes.xyxy[0][3])),
                Image.fromarray( result.plot()[:, :, [2,1,0]]))

################################################
# Main loop 
################################################
tmp_file = 'tmp.mp4'

st.title('FLL Experiment Evaluation')

################################################
# Configuration
################################################
with st.sidebar:
    run_detector = st.sidebar.radio('Experiment start / stop detection', ['Motion', 'Beep'], index=0)
    if run_detector == 'Beep':
        beep_freq = st.slider('Beep frequency', 100, 2000, 550)
        beep_power_threshold = st.slider('Beep power threshold', 0, 200, 100)
    obj_detection_conf = st.slider('Object detection confidence', 0.0, 1.0, 0.9)
    show_snapshots = st.checkbox('Show experiment start / stop snapshots')

################################################
# Video upload
################################################
video_file = st.file_uploader('Upload experiment video', type=['mp4'])
if video_file is not None:

    with open(tmp_file, 'wb') as f:
        f.write(video_file.getbuffer())
    
    clip = VideoFileClip(tmp_file)

    if run_detector == 'Beep':
        # Detect runs by beeps
        runs = detect_beeps(clip.audio, beep_freq, beep_power_threshold)
    else:
        # Motion detection
        runs = detect_motion(clip)    

    ################################################
    # Pick frames corresponing to detected runs start and stop times
    # and detect robots in these frames
    ################################################
    col_start, col_end = st.columns(2)

    experiments = pd.DataFrame(columns = ['X1', 'X2', 'dX', 'Y1', 'Y2', 'dY', 'T1', 'T2', 'dT'])

    for i, (t_start, t_end) in enumerate( runs ):

        # start and end frames
        start_frame = clip.get_frame(t_start)
        end_frame = clip.get_frame(t_end)

        # detect robot position
        start = robot_position( start_frame)
        end = robot_position( end_frame)
        if not start or not end:
            continue                               # no robot detected at either start or end frame => skip this experiment step

        if show_snapshots:
            col_start.image(start[2])
            col_end.image(end[2])

        experiments.loc[len(experiments.index)] = [
            start[0],
            end[0],
            abs(end[0] - start[0]),
            start[1],
            end[1],
            abs(end[1] - start[1]),
            t_start,
            t_end,
            t_end - t_start
        ]

    # display and save the experiments dataframe
    st.dataframe(experiments)

    csv = convert_df(experiments)
    st.download_button(
        'Download as CSV',
        csv,
        'experiment.csv',
        'text/csv',
        key='download-csv'
    )



