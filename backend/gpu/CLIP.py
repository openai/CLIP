import clip
import torch
import numpy as np
import cv2
from PIL import Image
import json

def embed(frames,model,preprocess,device):
    x = json.loads(frames)
    frames = np.array(x,dtype=np.uint8)
    print(frames.shape)
    frames_pil = [Image.fromarray(im_cv) for im_cv in frames]

    frames_preprocess = [preprocess(frame) for frame in frames_pil]
    frames_preprocess = torch.stack(frames_preprocess)
    print(frames_preprocess.shape)
    with torch.no_grad():
        vid_encodings = model.encode_image(frames_preprocess.to(device).half())
    print(vid_encodings.shape)
    vid_encodings_json = json.dumps(vid_encodings.numpy().tolist())
        
    return {'clip_encodings': vid_encodings_json}