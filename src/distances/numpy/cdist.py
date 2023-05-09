from distances import cdist
import numpy as np
from timeit import default_timer as timer
from scipy.spatial.distance import cdist as scdist

"""
print('creating array a...')
a = np.array([[2, 0],[0, 2]], dtype=np.float32)
print('creating array b...')
b = np.array([[0, 2],[2, 0]], dtype=np.float32)
print('creating array c...')
c = np.zeros((2, 2), dtype=np.float32)
"""


def cross_entropy(xa, xb):
    xc = np.zeros((xa.shape[0], xb.shape[0]), dtype=np.float32)
    for i, x in enumerate(xa):
        for j, y in enumerate(xb):
            ce = -np.mean(x * np.log(y + 1e-24))
            xc[i, j] += ce
    return xc


def binary_cross_entropy(xa, xb):
    xc = np.zeros((xa.shape[0], xb.shape[0]), dtype=np.float32)
    for i, x in enumerate(xa):
        for j, y in enumerate(xb):
            ce = -np.mean(x * np.log(y) + (1. - x) * np.log(1. - y + 1e-24))
            xc[i, j] += ce
    return xc


# for n_elements, n_rows, n_cols in np.random.randint(3, 5, (10, 3)):
n_rows = 256
n_cols = 256
n_elements = 2048

a = np.random.rand(n_rows, n_elements)
b = np.random.rand(n_cols, n_elements)

r = np.random.randint(0, 2, b.shape)
b = (1. - r) + (r * (np.random.randn(b.shape[0], b.shape[1]) * 1e-15))
b = b.astype(np.float32)
epsilon = 1e-33
#b = b - 1e-8
b = np.maximum(b, epsilon)


print('a', np.max(a), np.min(a))
print('b', np.max(b), np.min(b))

c = np.zeros((n_rows, n_cols), dtype=np.float32)
cdist(8, 'crossentropy', a.astype(np.float32), b.astype(np.float32), c)
xc = cross_entropy(a.astype(np.float32), b.astype(np.float32))
print('XC.shape: ', xc.shape)
print(np.max(np.abs(xc - c)))
print(np.mean(np.abs(xc - c)))
print(np.median(np.abs(xc - c)))

print('Computing binary cross entropy similarities...')
c = np.zeros((n_rows, n_cols), dtype=np.float32)
cdist(8, 'binarycrossentropy', a.astype(np.float32), b.astype(np.float32), c)
xc = binary_cross_entropy(a.astype(np.float32), b.astype(np.float32))
print(np.max(np.abs(xc - c)))
print(np.mean(np.abs(xc - c)))
print(np.median(np.abs(xc - c)))

print('Computing cityblock similarities...')
c = np.zeros((n_rows, n_cols), dtype=np.float32)
cdist(8, 'cityblock', a.astype(np.float32), b.astype(np.float32), c)
xc = scdist(a.astype(np.float32), b.astype(np.float32), 'cityblock').astype(np.float32)
print(np.max(np.abs(xc - c)))
print(np.mean(np.abs(xc - c)))
print(np.median(np.abs(xc - c)))

print('Computing euclidean similarities...')
c = np.zeros((n_rows, n_cols), dtype=np.float32)
cdist(8, 'euclidean', a.astype(np.float32), b.astype(np.float32), c)
xc = scdist(a.astype(np.float32), b.astype(np.float32), 'euclidean')
print(np.max(np.abs(xc - c)))
print(np.mean(np.abs(xc - c)))
print(np.median(np.abs(xc - c)))

print('Computing cosine similarities...')
c = np.zeros((n_rows, n_cols), dtype=np.float32)
cdist(8, 'cosine', a.astype(np.float32), b.astype(np.float32), c)
xc = scdist(a.astype(np.float32), b.astype(np.float32), 'cosine')
print(np.max(np.abs(xc - c)))
print(np.mean(np.abs(xc - c)))
print(np.median(np.abs(xc - c)))

"""
for metric in ["cityblock", "euclidean", "cosine", "crossentropy", "binarycrossentropy"]:
    try:
        start = timer()
        cdist(4, metric, a, b, c)
        stop = timer()
        print('time elapsed: {}\n'.format(stop - start))
    except:
        print('Error!')
"""

print("Done")
