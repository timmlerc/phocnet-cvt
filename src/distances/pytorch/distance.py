import torch
import numpy as np
from scipy.spatial.distance import cdist
import cpu.distance as cpu
import gpu.distance as gpu
from timeit import default_timer as Timer

def cityblock(XA, XB):

    XC = torch.zeros(XA.size(0), XB.size(0)).double()
    for x, xa in enumerate(XA):
        for y, xb in enumerate(XB):
            XC[x, y] = (xa - xb).abs().sum()
    return XC

#XA = torch.sigmoid(torch.rand((2048, 1024)) * 20. - 10.).double()
#XB = torch.sigmoid(torch.rand((2048, 1024)) * 20. - 10.).double()

XA = torch.rand((1, 1024))
XB = torch.rand((1, 1024))

print(XA.max(), XA.min(), XA.mean(), XA.var())

#XA = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).float()
#XB = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).float()
XC_sp = cdist(XA.numpy(), XB.numpy(), metric="cosine")

start = Timer()
XC_omp = cpu.cdist(XA, XB, 'cosine', 4)
stop = Timer()
print('Time: {}'.format(stop - start))
print('diff: {}'.format(np.mean(np.abs(XC_omp.cpu().numpy() - XC_sp))))

XA = XA.to(torch.device('cuda'))
XB = XB.to(torch.device('cuda'))
start = Timer()
XC_gpu = gpu.cdist(XA, XB, 'cosine')
stop = Timer()
print('Time: {}'.format(stop - start))

#print(XA.cpu().numpy()[:4,:4])
#print(XC_gpu.cpu().numpy()[:4,:4])
#print(XC_sp[:4,:4])

print(XA.cpu().numpy())
print(XC_gpu.cpu().numpy())
print(XC_sp)

print('diff: {}'.format(np.mean(np.abs(XC_gpu.cpu().numpy() - XC_sp))))

# XC = cityblock(XA.cpu(), XB.cpu())
# print(XC.cpu().numpy())
# print('diff: {}'.format(np.mean(np.abs(XC_gpu.cpu().numpy() - XC.cpu().numpy()))))
# print('diff: {}'.format(np.mean(np.abs(XC.cpu().numpy() - XC_sp))))

#print('diff: {}'.format(np.mean(np.abs(XC_omp.cpu().numpy() - XC_gpu.cpu().numpy()))))