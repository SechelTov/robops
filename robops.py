import streamlit as st
import numpy as np
from moviepy.editor import VideoFileClip
import math
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
from typing import Tuple
import pandas as pd

@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=True).encode('utf-8')

################################################
# Audio processing constants and functions
################################################
fps = 22000            # audio frames per second
cps = 10               # cuts per second
fpc = int(fps / cps)   # frames per cut
beep_freq_tolerance = 50

def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.

    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage :

        # calculating frequencies in ranges [400, 500] and [1000, 1100]
        # of a windowed signal sampled at 44100 Hz

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

cut = lambda i: clip.audio.subclip(i/cps,(i+1)/cps).to_soundarray(fps=22000, quantize=False)
#volume = lambda array: np.sqrt(((1.0*array)**2).mean())
samples = lambda array: array[:fpc][:, 0]   # take one column of audio signal
max_pwr = lambda array, freq: max([r[2] for r in goertzel( samples(array), fps, freq)[1]])

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
    beep_freq = st.slider('Beep frequency', 100, 2000, 550)
    beep_power_threshold = st.slider('Beep power threshold', 0, 200, 100)
    obj_detection_conf = st.slider('Object detection confidence', 0.0, 1.0, 0.9)

################################################
# Experiment video upload
################################################
video_file = st.file_uploader('Upload experiment video', type=['mp4'])
if video_file is not None:

    with open(tmp_file, 'wb') as f:
        f.write(video_file.getbuffer())
    
    clip = VideoFileClip(tmp_file)
    st.write(clip.duration)

    num_cuts = int(clip.duration*cps - 1)
    #cut_volumes = [volume(cut(i)) for i in range(0,int(clip.duration*cps - 1))]

    ################################################
    # Detect beeps
    # Same note designates BOTH an experiment start and end
    ################################################
    beep_pwrs  = [max_pwr(cut(i), (beep_freq - beep_freq_tolerance, beep_freq + beep_freq_tolerance)) for i in range(num_cuts)]
    
    beep_starts = [i 
                   for i, (cur, next) 
                   in enumerate(zip(beep_pwrs[:-1], beep_pwrs[1:])) 
                   if cur < beep_power_threshold and next > beep_power_threshold or i==0 and cur > beep_power_threshold]

    experiment_starts = beep_starts[0::2]
    experiment_ends = beep_starts[1::2]

    ################################################
    # Pick frames corresponing to beep timimg
    # and detect robots in these frames
    ################################################
    col_start, col_end = st.columns(2)

    experiments = pd.DataFrame(columns = ['X', 'Y', 'T'])

    for i, (s, e) in enumerate( zip(experiment_starts, experiment_ends) ):

        # time in seconds
        t_start = s / cps
        t_end = e / cps

        # start and end frames
        start_frame = clip.get_frame(t_start)
        end_frame = clip.get_frame(t_end)

        # detect robot position
        start = robot_position( start_frame)
        end = robot_position( end_frame)
        if not start or not end:
            continue                               # no robot detected at either start or end frame => skip this experiment step

        col_start.image(start[2])
        col_end.image(end[2])

        # transition
        transition = ( int(abs(end[0] - start[0])), int(abs(end[1] - start[1])) )

        experiments.loc[len(experiments.index)] = [transition[0], transition[1], t_end - t_start]

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




