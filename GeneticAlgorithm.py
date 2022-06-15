# flake8: noqa E501
import random
import numpy as np
from copy import deepcopy
import numpy as np
import cv2

try:
    from numba import njit
except ImportError:
    print("Numba not installed. Using slow Python version.")
    njit = lambda x: x

@njit
def affine_mat_to_mat_vec(affine_mat):
  mat = affine_mat[:, :2]
  vec = affine_mat[:, 2]
  return mat, vec

@njit
def mat_vec_to_affine(mat, vec):
  return np.concatenate((mat, vec.reshape((2, 1))), axis = 1)

@njit
def rotation(w):
  theta = np.random.uniform() * np.pi
  mat, vec = affine_mat_to_mat_vec(w)
  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
  mat = np.dot(rotation_matrix, mat)
  new_map = mat_vec_to_affine(mat, vec)
  w[:, :] = new_map

@njit
def scale(w):
  # probably need to change this
  s = np.random.uniform(0.5, 1.5)
  (a, b, e,
   c, d, f) = w.reshape((6,))
  if np.random.random() < 0.5:
    w[0, 0] = a * s
  else:
    w[0, 1] = b * s

@njit
def skew(w):
  s = np.random.uniform(0.0, 1.0)
  (a, b, e,
   c, d, f) = w.reshape((6,))
  if np.random.random() < 0.5:
    w[1, 0] = a * s + b
    w[1, 1] = c * s + d
  else:
    w[0, 0] = a + b * s
    w[0, 1] = c + d * s

@njit
def translation( w):
  (a, b, e,
   c, d, f) = w.reshape((6,))
  r = (e + f) / 4
  x = np.random.uniform(-r, r)
  w = w.copy()
  if np.random.random() < 0.5:
    w[0, 2] = e + x
  else:
    w[1, 2] = f + x


@njit
def map_mutation(chromosome):
    """
    mutate a chromosome by changing a random affine map
    """
    # needs repairing
    flag = np.random.random()
    i = np.random.randint(0, len(chromosome.genes))
    item = chromosome.genes[i]
    if flag < 0.25:   rotation(item)
    elif flag < 0.5:  translation(item)
    elif flag < 0.75: skew(item)
    else:             scale(item)

@njit
def map_perturbation(target, chromosome, STDCT, STDCP):
    i = np.random.randint(0, len(chromosome.genes))
    map = chromosome.genes[i]
    (a, b, e,
   c, d, f) = map.reshape((6,))
    randJitter = (1 - fitness(target, chromosome, STDCT, STDCP)) * np.random.uniform(-1., 1.)
    flag = np.random.random()
    if flag < 1 / 6 and np.abs(a + randJitter + e - 0.5) <= 0.5:   map[0,0] += randJitter
    elif flag < 2 / 6 and np.abs(b + randJitter + e - 0.5) <= 0.5: map[0,1] += randJitter
    elif flag < 3 / 6 and np.abs(c + randJitter + f - 0.5) <= 0.5: map[1,0] += randJitter
    elif flag < 4 / 6 and np.abs(d + randJitter + f - 0.5) <= 0.5: map[1,1] += randJitter
    elif flag < 5 / 6 and np.abs(e - 0.5) <= 0.5: map[0,2] += randJitter
    elif flag < 1 and np.abs(f - 0.5) <= 0.5: map[1,2] += randJitter

    if (map[0,0] + map[0,1] + map[0,2]) > 1 or (map[0,0] + map[0,1] + map[0,2]) < 0:
        map[0,0], map[0,1] = map[0,0] / 2, map[0,1] / 2
    if (map[1,0] + map[1,1] + map[1,2]) > 1 or (map[1,0] + map[1,1] + map[1,2]) < 0:
        map[1,0], map[1,1] = map[1,0] / 2, map[1,1] / 2


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
  return np.exp(-d**2/10000**2)

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


def contFactor(mat): return np.linalg.norm(mat[:,:2], ord=2)

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

def penalizeCompFac(mats, STDCP):
  return(np.exp(-(len(mats)/(2*STDCP))**2))

# the better the closer to 1
def fitness(target, chromo, STDCT, STDCP):
   mats = chromo.genes
   y = w(target, mats)
   return stacked_metric(target, y) * penalizeContFac(mats, STDCT) * penalizeCompFac(mats, STDCP) #+ punish_on_identity_map 




class Chromosome():
    def __init__(self, affines):
        self.genes = affines
    def __getitem__(self, i): return self.genes[i]
    def __len__(self, i): return len(self.genes)


class Population():
    def __init__(self, target, chromosomes, bestMax, chromLenMax, popMax, crossProb, mutProb, ifsMutProb, STDCT, STDCP):
        self.target = target
        self.members = chromosomes
        self.bestMax = bestMax
        self.chromLenMax = chromLenMax
        self.popMax = popMax
        self.crossProb = crossProb
        self.mutProb = mutProb
        self.ifsMutProb = ifsMutProb
        self.best = [Chromosome([])]
        self.STDCT = STDCT
        self.STDCP = STDCP

    def selection(self):
        memberFitBestSorted = sorted([(chromo, fitness(self.target, chromo, self.STDCT, self.STDCP)) for chromo in self.members], key=lambda x: x[1], reverse=True)
        self.best = memberFitBestSorted[:self.bestMax]

    def crossover(self):
        newMembers = []
        for parent1, _ in self.best:
            parent2, _ = random.choice(self.best)
            cPoint1 = random.randint(0, len(parent1))
            cPoint2 = random.randint(0, len(parent2))
            child1 = deepcopy(parent1.genes[0: cPoint1]) + deepcopy(parent2.genes[cPoint2:])
            child2 = deepcopy(parent2.genes[0: cPoint2]) + deepcopy(parent1.genes[cPoint1:])
            newMembers.extend([Chromosome(child1), Chromosome(child2)])
        self.members = newMembers


    def mutate(self):
        # need to mutate the best ones as well
        i = 0
        while i < len(self.members):
            if self.mutProb > np.random.random():
                if self.ifsMutProb > np.random.random():
                    if np.random.random() < 0.5 and len(self.members[i].genes) != 1:
                        self.members[i].genes.pop(npGetArrInd(self.members[i].genes, random.choice(self.members[i].genes)))
                    else:
                        self.members[i].genes.append(random_affine())
                else:
                    if np.random.random() < 0.5:
                        map_mutation(self.members[i])
                    else:
                        map_perturbation(self.target, self.members[i], self.STDCT, self.STDCP)
            i += 1

    def repair(self):
        #controls growth: stops chromosomes from having too many genes and the population being too large
        self.members.extend([i[0] for i in self.best])
        self.members = [i[0] for i in sorted([(chromo, fitness(self.target, chromo, self.STDCT, self.STDCP))
                                              for chromo in self.members], key=lambda x: x[1], reverse=True)]
        self.members = self.members[:self.popMax]
        self.members = [Chromosome(i.genes[:self.chromLenMax]) for i in self.members]


chromos = [Chromosome([random_affine() for _ in range(3)]) for _ in range(50)]
pop = Population(sierpinski, chromos, 15, 10, 50, 0.5, 0.1, 0.5, 0.5, 4)
for i in range(10000):
    pop.selection()
    pop.crossover()
    pop.mutate()
    pop.repair()

    if i % 1000 == 0:
        genes = pop.best[0][0].genes
        for mat in genes:
            print(mat)
        print(pop.best[0][1])

print("the best fitness value was " + str(pop.best[0][1]))
pic = np.ones((128,128), dtype = np.uint8)*255
genes = pop.best[0][0].genes
for mat in genes:
    print(mat)
for _ in range(7):
    pic = w(pic, genes)
cv2.imshow("result", pic)
cv2.waitKey(0)
print(stacked_metric(maple_leaf.maple_leaf_white_0_1, pic))