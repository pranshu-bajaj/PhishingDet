"""
code written by Harshit Batra 
"""

import random
from turtle import position
import numpy
import math
from solution import solution
import time


def MROM(objf, lb, ub, dim, SearchAgents_no, Max_iter,trainInput,trainOutput,net, m=7 ):

    
    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5

    # initialize Xw ,Xavg and Xb
    Xw_pos = numpy.zeros(dim)
    Xw_score = float("-inf")

    Xavg_pos = numpy.zeros(dim)
    Xavg_score = float("inf")

    xb_pos = numpy.zeros(dim)
    best_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i])

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('MROM is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, Max_iter):
        index = 0 # index of the worst individual
        for i in range(0, SearchAgents_no):

            Xavg_pos = Positions.mean(0)

            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i, :] = numpy.clip(Positions[i, :], lb, ub)

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :],trainInput,trainOutput,net)
            Xavg_score = objf(Xavg_pos,trainInput,trainOutput,net)
            # Update Xw
            if fitness > Xw_score:
                Xw_score = fitness
                Xw_pos = Positions[i, :].copy()
                index = i
            if fitness < best_score:
                best_score = fitness # update best score
                xb_pos = Positions[i, :].copy() # update best position    
        
        # Update Xw basewd on xavg
        if Xavg_score < Xw_score:
            Xw_score = Xavg_score
            Xw_pos = Xavg_pos.copy()
            Positions[index, :] = Xw_pos.copy()


        # Update the Position of search agents 
        #phase 1 
        for i in range(0, SearchAgents_no):
            j = numpy.random.randint(0, SearchAgents_no-1) # j!=1
            if j>= i :
                j = j+1
            fitness_i = objf(Positions[i, :],trainInput,trainOutput,net)
            fitness_j = objf(Positions[j, :],trainInput,trainOutput,net)
            if fitness_i > fitness_j:
                if fitness_i > Xavg_score:
                    x_worst = Positions[i, :].copy()
                    if fitness_j > Xavg_score:
                        x_best = Xavg_pos.copy()
                        x_medium = Positions[j, :].copy()
                    else:
                        x_best = Positions[j, :].copy()
                        x_medium = Xavg_pos.copy()
                else :
                    x_worst = Xavg_pos.copy()
                    x_best = Positions[j, :].copy()
                    x_medium = Positions[i, :].copy()
            else:
                if fitness_j > Xavg_score:
                    x_worst = Positions[j, :].copy()
                    if fitness_i > Xavg_score:
                        x_best = Xavg_pos.copy()
                        x_medium = Positions[i, :].copy()
                    else:
                        x_best = Positions[i, :].copy()
                        x_medium = Xavg_pos.copy()
                else :
                    x_worst = Xavg_pos.copy()
                    x_best = Positions[i, :].copy()
                    x_medium = Positions[j, :].copy()
            # Update the Position of search agents 
            Xt = numpy.zeros(dim)

            
            Xt = x_medium-x_worst #eq 1 in the paper
            
            T = i/Max_iter

            metal_ratio = ( m + abs(math.sqrt(m**2 + 4)) ) / 2 # Metal ratio 

            ft = metal_ratio*(metal_ratio**T - (m-metal_ratio)**T) / abs(math.sqrt(m**2 + 4)) # eq 2 in the paper

            a = numpy.random.rand()

            if a < 0.5 :
                Xnew = (1-ft)*x_best + numpy.random.rand()*ft*Xt # eq 3 in the paper
            if a >= 0.5:
                Xnew = (1-ft)*(x_best-x_medium) + 2*numpy.random.rand()*ft*Xt
            if objf(Xnew,trainInput,trainOutput,net) < objf(Positions[i, :],trainInput,trainOutput,net):
                Positions[i, :] = Xnew.copy()
                Positions[i, :] = numpy.clip(Positions[i, :], lb, ub)  # eq 4 in the paper
            
        #phase 2
        for i in range(0, SearchAgents_no):

            # Update the Position of search agents
            b = numpy.random.rand() 
            if b < 0.5 :
                Xnew = Positions[i,:] + (1/metal_ratio)*numpy.random.rand()*(xb_pos-Xw_pos) # eq 5 in the paper
            if b >= 0.5 :
                Xnew = (Positions[i,:]-xb_pos)+ (1/metal_ratio)*numpy.random.rand()*(Positions.mean(0)-Xw_pos) 

            if objf(Xnew,trainInput,trainOutput,net) < objf(Positions[i, :],trainInput,trainOutput,net):
                Positions[i, :] = Xnew.copy()
                
        Convergence_curve[l] = best_score

        if l % 1 == 0:
            print(
            ["At iteration " + str(l) + " the best fitness is " + str(best_score)]
            )
    print(len(Convergence_curve))
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "MROM"
    s.objfname = objf.__name__
    s.best = best_score
    s.bestIndividual = xb_pos

    return s