from horovod import torch as hvd
import torch.nn as nn
import torch

print("start hvd")
hvd.init()
n_gpu = hvd.size()
device = torch.device("cuda", hvd.local_rank())
torch.cuda.set_device(hvd.local_rank())
print("hvd init")
rank = hvd.rank()

print("device: {} n_gpu: {}, rank: {}, "
            .format(
                device, n_gpu, hvd.rank()))


a=nn.Linear(3,4).cuda()