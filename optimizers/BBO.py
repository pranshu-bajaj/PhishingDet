"""
code by Harshit Batra parts of code taken from evolopy library by hossam faris
"""
import random
import numpy
import math
from solution import solution
import time


def BBO(objf, lb, ub, dim, SearchAgents_no, Max_iter,trainInput,trainOutput,net):

    # dim=30
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500

    Basket_Location = numpy.zeros(dim)
    Basket_fitness = float("inf") # change this to -inf for maximization problems
    Best_player_Location = numpy.zeros(dim)
    Best_player_Energy = float("inf")
    Opposite_basket_location = numpy.zeros(dim)
    Opposite_Basket_fitness = float("-inf") 


    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)

    # Initialize the locations of Players on basket ball court 
    X = numpy.asarray(
        [x * (ub - lb) + lb for x in numpy.random.uniform(0, 1, (SearchAgents_no, dim))]
    )

    # Initialize convergence
    convergence_curve = numpy.zeros(Max_iter)

    ############################
    s = solution()

    print('BBO is now tackling  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Check boundries

            X[i, :] = numpy.clip(X[i, :], lb, ub)

            # fitness of locations
            fitness = objf(X[i, :],trainInput,trainOutput,net)

            # Update the location of Basket
            if fitness > Opposite_Basket_fitness:  # Change this to > for maximization problem
                Opposite_Basket_fitness = fitness
                Opposite_basket_location = X[i, :]

            if Basket_fitness< fitness < Best_player_Energy:  # Change this to > for maximization problem
                Best_player_Location = X[i, :]
                Best_player_Energy = fitness

            if fitness < Basket_fitness :  # Change this to > for maximization problem
                Best_player_Location = Basket_Location.copy()
                Best_player_Energy = Basket_fitness
                Basket_fitness = fitness
                Basket_Location = X[i, :].copy()

        Tuning_parameter = 2 * (1 - (t / Max_iter)) 

        # Update the location of search agents
        if t < Max_iter / 2:
            for i in range(0, SearchAgents_no):
                    # -------------- Phase 1  -------------------
                R1 = random.random()
                Random_player_index = numpy.random.randint(0, SearchAgents_no-1) # j!=1
                if Random_player_index>= i : # Random_player_index!=i
                    Random_player_index = Random_player_index+1
                X_rand = X[Random_player_index, :]
                if R1 < 0.33:
                    # pass ball to random player 
                    X[i, :] = X_rand - random.random() * abs(2 * random.random()*X_rand - 2 * random.random() * X[i, :])

                elif 0.33 <=R1 < 0.67:
                    # Movement towards the Basket 
                    X1 = X[i, :] + random.random()*(Basket_Location-Opposite_basket_location)
                else :
                    X[i, :] = (Basket_Location - X.mean(0)) - random.random() * ((ub - lb) * random.random() + lb)
                

                X[i, :] = numpy.clip(X[i, :], lb, ub)        
        for i in range(0, SearchAgents_no,5):
            for j in range(0 , 5) :
                X_best_fitness = float("inf")
                X_worst_fitness = float("-inf")
                fitness = objf(X[i+j, :],trainInput,trainOutput,net)
                if fitness > X_worst_fitness:
                    X_worst_fitness = fitness
                    X_worst = X[i+j, :].copy()
                if fitness < X_best_fitness :  # Change this to > for maximization problem
                    X_best_fitness = fitness
                    X_best = X[i+j, :].copy()

            player =[0 ,1 , 2 , 3 , 4 ]

            for k in range(0 , 5) :
                j = random.choice(player)
                if j == 0 :
                    X1 = Best_player_Location - Tuning_parameter*abs(random.random()*Best_player_Location-X[i+k, :])
                    if objf(X1,trainInput,trainOutput,net) < fitness:  # improved move?
                        X[i+k, :] = X1.copy()


                elif j == 1 :
                    X1 = Basket_Location + Tuning_parameter*(Basket_Location-random.random()*X[i+k, :])
                    if objf(X1,trainInput,trainOutput,net) < fitness:  # improved move?
                        X[i+k, :] = X1.copy()

                
                elif j == 2 :
                    X1 = Basket_Location + Tuning_parameter*(Basket_Location-random.random()*X[i+k, :])+ numpy.multiply(numpy.random.randn(dim), Levy(dim))
                    if objf(X1,trainInput,trainOutput,net) < fitness:  # improved move?
                        X[i+k, :] = X1.copy()

                elif j == 3 :
                    X1 = (X_best-X_worst) + Tuning_parameter*(X_best-random.random()*X[i+k, :])
                    if objf(X1,trainInput,trainOutput,net) < fitness:  # improved move?
                        X[i+k, :] = X1.copy()

                elif j == 4 :
                    X1 = Basket_Location + Tuning_parameter*(Basket_Location-X[i+k, :])
                    if objf(X1,trainInput,trainOutput,net) < fitness:  # improved move?
                        X[i+k, :] = X1.copy()

        convergence_curve[t] = Basket_fitness
        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(Basket_fitness)
                ]
            )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "BBO"
    s.objfname = objf.__name__
    s.best = Basket_fitness
    s.bestIndividual = Basket_Location

    return s


def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step
