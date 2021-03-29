import numpy as np
from scipy.stats import truncnorm
import PIL.ImageDraw
import PIL.ImageFont


def truncated_z_sample(batch_size, dim_z, truncation=1):
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z))
    return truncation * values

def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        imarray = np.uint8(imarray)
        # raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows * H, cols * W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid

def annotate_outscore(array, outscore):
    for i in range(array.shape[0]):
        I = PIL.Image.fromarray(np.uint8(array[i,:,:,:]))
        draw = PIL.ImageDraw.Draw(I)
        font =  PIL.ImageFont.truetype("/data/scratch/swamiviv/projects/stylegan2-ada-pytorch/clip_steering/arial.ttf", int(array.shape[1]/8.5))
        message = str(round(np.squeeze(outscore)[i], 2))
        x, y = (0, 0)
        w, h = font.getsize(message)
        #print(w, h)
        draw.rectangle((x, y, x + w, y + h), fill='white')
        draw.text((x, y), message, fill="black", font=font)
        array[i, :, :, :] = np.array(I)
    return(array)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)