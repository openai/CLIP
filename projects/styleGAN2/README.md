0. Install deps to run clip and stylegan2-ada-pytorch in a python virtual env.
1. Install deps from stylegan2-ada-pytorch repo. Major ones are pytorch >= 1.7 and CUDA >= 11.0
2. Download ffhq-pretrained stylegan2 model from the above repo.
3. Use the virtual environment from above and Run through generation_demo.ipynb - this code samples images from
a styleGAN2 network and scores them using CLIP
4. ganalyze_with_clip.py is the main code that runs the steering pipeline with a generative model and CLIP. Change output
paths and model paths from within the code.