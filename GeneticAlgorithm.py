# flake8: noqa E501
import random
import numpy as np
from copy import deepcopy
import cv2
from mutations import map_mutation, map_perturbation
from utils import w, sierpinski, random_affine
from fitness import *
from sortedcontainers import SortedDict, SortedKeyList

try:
    from numba import njit
except ImportError as e:
    print(e)
    print("Numba not installed. Using slow Python version.")
    njit = lambda x: x

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
        self.best = [[]]
        self.STDCT = STDCT
        self.STDCP = STDCP
        self.smallest_fitness = 0

    def selection(self):
        # already sorted in repair
        # sorted(self.members, 
        #         key=lambda x: fitness(self.target, x, self.STDCT, self.STDCP),
        #         reverse=True)
        # descending order
        self.best = self.members[:self.bestMax]
        self.smallest_fitness = fitness(self.target, self.best[0], self.STDCT, self.STDCP)

    def crossover(self):
        newMembers = []
        for parent1 in self.best:
            parent2 = random.choice(self.best)
            cPoint1 = random.randint(1, len(parent1))
            cPoint2 = random.randint(1, len(parent2))
            child1 = deepcopy(parent1[0: cPoint1]) + deepcopy(parent2[cPoint2:])
            child2 = deepcopy(parent2[0: cPoint2]) + deepcopy(parent1[cPoint1:])
            newMembers.extend([child1, child2])
        self.members = newMembers


    def mutate(self):
      self.members.extend(deepcopy(self.best))
      for chromo in self.members:
        if self.mutProb > np.random.random():
            if self.ifsMutProb > np.random.random(): # ifs mutation
                if np.random.random() < 0.5 and len(chromo) != 1:
                    chromo.pop(np.random.randint(0, len(chromo)))
                else: chromo.append(random_affine())
            else: # map mutation
                if np.random.random() < 0.5: map_mutation(chromo)
                else: map_perturbation(self.target, chromo, self.STDCT, self.STDCP)

    def repair(self):
        # remove bad chromosomes
        self.members.extend(self.best)
        self.members.sort(key=lambda x: fitness(self.target, x, self.STDCT, self.STDCP),
            reverse=True)
        self.members = self.members[:self.popMax]
        #controls growth: stops chromosomes from having too many genes and the population being too large
        self.members = [i[:self.chromLenMax] for i in self.members]


chromos = [[random_affine() for _ in range(3)] for _ in range(50)]
pop = Population(
    target=sierpinski,
    chromosomes=chromos,
    bestMax=15,
    chromLenMax=7,
    popMax=100,
    crossProb=0.5,
    mutProb=0.5,
    ifsMutProb=0.5,
    STDCT=0.5,
    STDCP=2
)
for j in range(1000):
    for i in range(50000):
        pop.selection()
        pop.crossover()
        pop.mutate()
        pop.repair()

        if i % 1000 == 0:
            for chromo in pop.members[::10]:
                print(fitness(pop.target, chromo, pop.STDCT, pop.STDCP))
            genes = pop.best[0]
            for mat in genes:
                print(mat)
            f = fitness(pop.target, genes, pop.STDCT, pop.STDCP)
            d = stacked_metric(pop.target, w(pop.target, genes))
            p_k = penalizeCompFac(genes, pop.STDCP)
            p_c = penalizeContFac(genes, pop.STDCT)
            print(f, d, p_k, p_c)
            pic = np.ones((64,64), dtype = np.uint8)*255
            for _ in range(6):
                pic = w(pic, genes)
            cv2.imwrite(f"result_{j}_{i // 1000}.png", pic)

print("the best fitness value was " + str(pop.best[0]))
pic = np.ones((64,64), dtype = np.uint8)*255
genes = pop.best[0]
for mat in genes:
    print(mat)
for _ in range(6):
    pic = w(pic, genes)
cv2.imshow("result", pic)
cv2.waitKey(0)
# print(stacked_metric(maple_leaf.maple_leaf_white_0_1, pic))