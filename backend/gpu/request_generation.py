import requests
import json
import numpy as np

num_frames = 1
shape = (num_frames, 300, 300, 3)

random_frames = np.random.randint(0,255,size=shape,dtype = np.uint8)
fr_json = json.dumps(random_frames.tolist())

r = requests.post('http://127.0.0.1:8000/', json={"video_frames": fr_json})
print(r.content)