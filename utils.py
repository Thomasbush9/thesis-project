import numpy as np
import einops 
from tqdm import tqdm 
from pathlib import Path
import cv2


def captureVideo(video_path:str)->np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # For grayscale
        frames.append(frame)
    cap.release()
    return  np.array(frames) 

def loadVideoArray(video_path:str | list )->np.ndarray| dict:
    '''
    Returns dict of np.arrays per video (if one video it returns one array)
    '''
    
    if isinstance(video_path, str): #single video
        return captureVideo(video_path)

    
    if isinstance(video_path, list):
        
        return {video:captureVideo(video) for video in tqdm(video_path, desc='Loading videos')}



