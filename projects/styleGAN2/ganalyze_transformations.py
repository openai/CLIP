import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.io as sio

class OneDirection(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(OneDirection, self).__init__()
        print("\napproach: ", "one_direction\n")
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.w = nn.Parameter(torch.randn(1, self.dim_z))
        self.criterion = nn.MSELoss()


    def transform(self,z,y,step_sizes,**kwargs):
        if y is not None:
            assert(len(y) == z.shape[0])

        interim = step_sizes * self.w

        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()

        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss

class ClassDependent(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(ClassDependent, self).__init__()
        print("\napproach: ", "class_dependent\n")
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.NN_output = nn.Linear(self.vocab_size, self.dim_z)
        self.criterion = nn.MSELoss()


    def transform(self,z,y,step_sizes,**kwargs):
        assert (y is not None)
        interim = step_sizes * self.NN_output(y)
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss