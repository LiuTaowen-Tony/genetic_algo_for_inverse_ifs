import numpy as np
from utils import *
try: from numba import njit
except ImportError as e:
  print(e)
  print("Numba not installed. Using slow Python version.")
  njit = lambda x: x

@njit
def contFactor(mat): return np.linalg.norm(mat[:,:2], ord=2)

@njit
def penalizeContFac(mats, STDCT):
  matContFactors = [contFactor(mat) for mat in mats]
  maxCFac = max(matContFactors)
  while maxCFac > 1:
    badMatIndex = matContFactors.index(maxCFac)
    (a, b, e,
     c, d, f) = mats[badMatIndex].reshape((6,))
    opNorm = np.linalg.norm(np.array([[a, b],
                                      [c, d]]), ord=2)
    mats[badMatIndex] = np.array([[a/opNorm, b/opNorm, e],
                                 [c/opNorm, d/opNorm, f]])
    matContFactors = [contFactor(mat) for mat in mats]
    maxCFac = max(matContFactors)
  return (1 - maxCFac**10) * np.exp(-(maxCFac/(2*STDCT))**2)

@njit
def penalizeCompFac(mats, STDCP):
  return(np.exp(-(len(mats)/(2*STDCP))**2))

def stacked_metric(pic1, pic2):
  d = 0
  size = pic1.shape[0]
  factor = 1
  while size >= 8:
    d_t = np.sum(np.square(pic1 - pic2))
    d += d_t * factor
    factor = 4 * factor ** 2
    size = int(size / 2)
    pic1 = cv2.resize(pic1, (size, size))
    pic2 = cv2.resize(pic2, (size, size))
  return d

# the better the closer to 1
def fitness(target, mats, STDCT, STDCP):
   y = w(target, mats)
   d = stacked_metric(target, y)
  #  punish_identity = 
  #  for mat in mats:

   return np.exp(-d**2/20000**2) * penalizeContFac(mats, STDCT) * penalizeCompFac(mats, STDCP) #+ punish_on_identity_map 


