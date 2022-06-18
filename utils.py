import numpy as np
import cv2

def w(x, matList):
  x_l, y_l = x.shape
  result = np.zeros((x_l, y_l))
  for mat in matList:
    mat = mat.copy()
    mat[0,2] = mat[0,2] * x_l
    mat[1,2] = mat[1,2] * y_l
    result += cv2.warpAffine(x, mat, x.shape)
  return result

# get sierpinski's image
def sierpinski():
  pic = np.ones((64,64), dtype = np.float32)
  mat1 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0]])
  mat2 = np.array([[0.5, 0, 0.5], 
                   [0, 0.5, 0]])
  mat3 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0.5]])
  for _ in range(100):
    pic = w(pic, [mat1, mat2, mat3])
  return pic

sierpinski = sierpinski()

def fern():
  pic = np.ones((128,128), dtype = np.float32)
  mat1 = np.array([[0, 0, 0], 
                   [0.16, 0, 0]])
  mat2 = np.array([[0.85, 0.04, 0], 
                   [-0.04, 0.85, 1.6]])
  mat3 = np.array([[0.2, -0.26, 0], 
                   [0.23, 0.22, 1.6]])
  mat4 = np.array([[-0.15, 0.28, 0], 
                   [0.26, 0.24, 0.44]])
  for _ in range(10):
    pic = w(pic, [mat1, mat2, mat3, mat4])
  return pic

fern = fern()

def random_affine() -> np.ndarray:
    e, f = np.random.uniform(0., 1., 2)
    a, b = np.random.uniform(-e, 1-e, 2)
    if (a + b + e) > 1 or (a + b + e) < 0:
        a, b = a / 2, b / 2
    c, d = np.random.uniform(-f, 1-f, 2)
    if (c + d + f) > 1 or (c + d + f) < 0:
        c, d = c / 2, d / 2
    return np.array([[a, b, e],
                     [c, d, f]])

def det_abs(mat):
  return np.abs(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])

cv2.imshow("pic", fern)
cv2.waitKey(0)