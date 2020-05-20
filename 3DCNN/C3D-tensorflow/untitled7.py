import cv2
from cv2 import DualTVL1OpticalFlow_create as DualTVL1
import numpy as np
_IMAGE_SIZE = 224
import os
 
_EXT = ['.avi', '.mp4']


def _video_length(video_path):
    video = cv2.VideoCapture(video_path)
    total = 0
    count = 0
    while True:
        video.set(cv2.CAP_PROP_POS_MSEC, (count * 100))
        (grabbed, frame) = video.read()
        if not grabbed:
            break
        total += 1
        count +=1
    return total
#
#def _video_length(video_path):
#    _, ext = os.path.splitext(video_path)
#    if not ext in _EXT:
#        raise ValueError('Extension "%s" not supported' % ext)
#    cap = cv2.VideoCapture(video_path)     
#    if not cap.isOpened():
#        raise ValueError("Could not open the file.\n{}".format(video_path))
#    if cv2.__version__ >= '3.0.0':
#        cap.set(cv2.CAP_PROP_POS_MSEC, (1000))
#        print("fvefd")
#        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
#    else:
#        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
#    length = int(cap.get(CAP_PROP_FRAME_COUNT))
#    return length

def compute_rgb(video_path):
    cap = cv2.VideoCapture(video_path)
     
    rgb = []
    vid_len = _video_length(video_path)
    count = 0
    for _ in range(vid_len-1):
        cap.set(cv2.CAP_PROP_POS_MSEC, (count * 100))
        
        ret, frame2 = cap.read()
        if(frame2==None).any():
            continue
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
        curr = np.array(curr,dtype = np.float64)
        curr = 2*(curr - np.min(curr))/np.ptp(curr)-1
        rgb.append(curr)
        count+=1
    cap.release()
    rgb = np.array(rgb, dtype = np.float64)
    return rgb

def compute_TVL1(video_path):
    TVL1 = DualTVL1()
    cap = cv2.VideoCapture(video_path)
    
    ret, frame1 = cap.read()
    if(frame1==None).any():
        return
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (_IMAGE_SIZE, _IMAGE_SIZE))
    flow = []
    vid_len = _video_length(video_path)
    count = 0
    for _ in range(vid_len-1):
        cap.set(cv2.CAP_PROP_POS_MSEC, (count * 100))
        ret, frame2 = cap.read()
        if(frame2==None).any():
            continue
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
        curr_flow = TVL1.calc(prev, curr, None)
        assert(curr_flow.dtype == np.float32)
        curr_flow[curr_flow >= 20] = 20
        curr_flow[curr_flow <= -20] = -20
        curr_flow = 2*(curr_flow - np.min(curr_flow))/np.ptp(curr_flow)-1
        flow.append(curr_flow)
        prev = curr
        count +=1
    cap.release()
    flow = np.array(flow)
    return flow

print(compute_rgb('D:/videos/[2Sec] Cute.mp4').shape)
#print(compute_rgb('D:/autism/Dataset/hmdb51_org/brush_hair/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi'))

