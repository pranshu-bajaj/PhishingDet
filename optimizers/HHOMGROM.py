# -*- coding: utf-8 -*-
"""
code by Harshit Batra parts of code taken from evolopy library by hossam faris 

"""
import random
import numpy
import math
from solution import solution
import time


def HHOMGROM(objf, lb, ub, dim, SearchAgents_no, Max_iter,trainInput,trainOutput,net):

    # dim=30
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500

    # initialize the location and Energy of the rabbit
    Xw_pos = numpy.zeros(dim)
    Xw_score = float("-inf")

    Xavg_pos = numpy.zeros(dim)
    Xavg_score = float("inf")

    Rabbit_Location = numpy.zeros(dim)
    Rabbit_Energy = float("inf")  # change this to -inf for maximization problems

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)

    # Initialize the locations of Harris' hawks
    X = numpy.asarray(
        [x * (ub - lb) + lb for x in numpy.random.uniform(0, 1, (SearchAgents_no, dim))]
    )

    # Initialize convergence
    convergence_curve = numpy.zeros(Max_iter)

    ############################
    s = solution()

    print('HHO is now tackling  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Check boundries

            X[i, :] = numpy.clip(X[i, :], lb, ub)
            
            Xavg_pos = X.mean(0)
            Xavg_score = objf(Xavg_pos,trainInput,trainOutput,net)
            # fitness of locations
            fitness = objf(X[i, :],trainInput,trainOutput,net)


            # Update the location of Rabbit
            if fitness < Rabbit_Energy:  # Change this to > for maximization problem
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()
            if fitness > Xw_score:
                Xw_score = fitness
                Xw_pos = X[i, :].copy()

        E1 = 2 * (1 - (t / Max_iter))  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, SearchAgents_no):

            E0 = 2 * random.random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (
                E0
            )  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    X[i, :] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * X[i, :]
                    )

                elif q >= 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                j = numpy.random.randint(0, SearchAgents_no-1) # j!=1
                if j>= i :
                    j = j+1
                fitness_i = objf(X[i, :])
                fitness_j = objf(X[j, :])
                if fitness_i > fitness_j:
                    if fitness_i > Xavg_score:
                        x_worst = X[i, :].copy()
                        if fitness_j > Xavg_score:
                            x_best = Xavg_pos.copy()
                            x_medium = X[j, :].copy()
                        else:
                            x_best = X[j, :].copy()
                            x_medium = Xavg_pos.copy()
                    else :
                        x_worst = Xavg_pos.copy()
                        x_best = X[j, :].copy()
                        x_medium = X[i, :].copy()
                else:
                    if fitness_j > Xavg_score:
                        x_worst = X[j, :].copy()
                        if fitness_i > Xavg_score:
                            x_best = Xavg_pos.copy()
                            x_medium = X[i, :].copy()
                        else:
                            x_best =   X[i, :].copy()
                            x_medium = Xavg_pos.copy()
                    else :
                        x_worst = Xavg_pos.copy()
                        x_best = X[i, :].copy()
                        x_medium = X[j, :].copy()
                # Update the Position of search agents
                #phase 1 
                Xt = numpy.zeros(dim)

            
                Xt = x_medium-x_worst #eq 1 in the paper
                
                T = i/Max_iter

                phi = ( 1 + abs(math.sqrt(5)) ) / 2 # goldern ratio 

                ft = phi*(phi**T - (1-phi)**T) / abs(math.sqrt(5)) # eq 2 in the paper
                a = numpy.random.rand()
                if a>=0.5:
                    Xnew = (1-ft)*x_best + 2*numpy.random.rand()*ft*Xt # eq 3 in the paper
                if a<0.5:
                    Xnew = (1-ft)*(x_best-x_medium) + 2*numpy.random.rand()*ft*Xt

                if objf(Xnew) < objf(X[i, :]):
                    X[i, :] = Xnew.copy()
                    X[i, :] = numpy.clip(X[i, :], lb, ub)  # eq 4 in the paper
            
        if abs(Escaping_Energy) < 1:
            for i in range(0, SearchAgents_no):
                    # Update the Position of search agents
                a = numpy.random.rand()
                if a>=0.5:
                    Xnew = X[i,:] + (1/phi)*numpy.random.rand()*(Rabbit_Location-Xw_pos) # eq 5 in the paper
                if a<=0.5:
                    Xnew = (X[i,:]-Rabbit_Location)+ (1/phi)*numpy.random.rand()*(X.mean(0)-Xw_pos) 
                if objf(Xnew) < objf(X[i, :]):
                    X[i, :] = Xnew.copy() 

                

        convergence_curve[t] = Rabbit_Energy
        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(Rabbit_Energy)
                ]
            )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHOMGROM"
    s.objfname = objf.__name__
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location

    return s
