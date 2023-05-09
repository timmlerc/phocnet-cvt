import torch
import numpy
from scipy.spatial.distance import cdist
import distance as cuda
from timeit import default_timer as Timer

def cityblock(XA, XB):

    XC = list()
    for xa in XA:
        xa = xa.unsqueeze(0).expand_as(XB)
        XC.append( (xa-XB).abs().sum(0) )
    return torch.stack(XC)

XA = torch.randn(4, 4)
XB = torch.randn(4, 4)
XC = torch.zeros((XA.size(0), XB.size(0)))

print(XA.type())
print(XB.type())
print(XC.type())

start = Timer()
XC = cuda.cdist(XA.to(torch.device('cuda')), XB.to(torch.device('cuda')), "cityblock")
stop = Timer()
print('Time: {}'.format(stop - start))

start = Timer()
XD = cdist(XA.cpu().numpy(), XB.cpu().numpy(), metric="cityblock")
stop = Timer()
print('Time: {}'.format(stop - start))

start = Timer()
XE = cityblock(XA.cpu(), XB.cpu())
stop = Timer()
print('Time: {}'.format(stop - start))

print('diff: {}'.format(numpy.mean(numpy.abs(XC.cpu().numpy() - XD))))
print('diff: {}'.format( numpy.mean( numpy.abs( XC.cpu().numpy() - XE.numpy() ) ) ))
print('diff: {}'.format( numpy.mean( numpy.abs(XD - XE.numpy()) ) ) )

print(XA.cpu().float().numpy())
print(XB.cpu().float().numpy())
print(XC.cpu().float().numpy())
print(XD.astype(numpy.float32))
print(XE.cpu().float().numpy())