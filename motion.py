import numpy as np 
from moviepy.editor import VideoFileClip, VideoClip, clips_array
import cv2

# Utility to iterate over pairs of consequtive items of an iterator
# https://stackoverflow.com/questions/54470101/how-do-i-refer-to-a-value-from-a-previous-iteration
#
def prev_and_next(itr, preprocess=None):
    prev = None
    for next in itr:
        if preprocess: next = preprocess(next)
        if prev is not None:
            yield prev, next
        prev = next

# Utility to find a non-zero blocks (sequences of items) in a list
# https://stackoverflow.com/questions/31544129/extract-separate-non-zero-blocks-from-array
#
import itertools
import operator

def blocks(L, threshold=1):
    L_ =  [threshold if v>=threshold else 0 for v in L]
    return [[i for i,value in it] for key,it in itertools.groupby(enumerate(L_), key=operator.itemgetter(1)) if key != 0]

# Compare between frames
# Returns a frame of differences between the two frames
#
def diff( f1, f2, threshold=20):

    def prep_frame(img):
        f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(src=f, ksize=(5,5), sigmaX=0)

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff( src1=prep_frame(f1), src2=prep_frame(f2))

    # dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # only take areas that are different enough
    return cv2.threshold(src=diff_frame, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)[1]


################################################
# Detect runs start / stop times by motion
# Take odd motions and ignore even ones as these are returns to initial position
################################################
def detect_motion( videoclip ):

    # count amount of white (i.e. moving) pixels in each frame
    moving_pixels = [np.sum(diff(f1,f2) == 255) for f1, f2 in prev_and_next( videoclip.iter_frames() )]
    
    # detect block of frames with many (>1000) moving pixels
    bb = blocks(moving_pixels, threshold=1000)
    
    # take start and stop indexes of odd blocks
    # ignore small blocks of length <= 3
    # convert frames into seconds
    # return list of tuples of (experiment_start, experiment_end)
    return [(b[0] / videoclip.fps, b[-1] / videoclip.fps) for b in bb if len(b)>3][::2]