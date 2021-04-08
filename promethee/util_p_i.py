###############################################################################

# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os

###############################################################################

# Function: Target
def target_function():
    return

# Function: Clip Thresholds
def clip_thresholds(population, indv = 0, size = 3, min_values = [-5,-5], max_values = [5,5]):
    size = size
    for i in range(0, size):
        if (population[indv][size*1 + i] > population[indv][size*2 + i]):
            population[indv][size*1 + i] = population[indv][size*2 + i]
    population[indv,:-1] = np.clip(population[indv,:-1], min_values, max_values)
    return population

# Function: Initialize Variables
def initial_population(population_size = 5, size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((population_size, len(min_values) + 1))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j]) 
        population = clip_thresholds(population, indv = i, size = size, min_values = min_values, max_values = max_values)
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
    return population

# Function: Fitness
def fitness_function(population): 
    fitness = np.zeros((population.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ population[i,-1] + abs(population[:,-1].min()))
    fit_sum = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], size = 3, mu = 1, elite = 0, target_function = target_function):
    offspring = np.copy(population)
    b_offspring = 0
    if (elite > 0):
        preserve = np.copy(population[population[:,-1].argsort()])
        for i in range(0, elite):
            for j in range(0, offspring.shape[1]):
                offspring[i,j] = preserve[i,j]
    for i in range (elite, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - 1):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        offspring = clip_thresholds(offspring, indv = i, size = size, min_values = min_values, max_values = max_values)
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1]) 
    return offspring
 
# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])
        offspring = clip_thresholds(offspring, indv = i, size = size, min_values = min_values, max_values = max_values)
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1])                        
    return offspring

# Function: GA
def genetic_algorithm(population_size = 5, mutation_rate = 0.1, elite = 0, min_values = [-5,-5], max_values = [5,5], eta = 1, mu = 1, generations = 50, size = 3, target_function = target_function):    
    count = 0
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, size = size, target_function = target_function)
    fitness = fitness_function(population)    
    elite_ind = np.copy(population[population[:,-1].argsort()][0,:])
    while (count <= generations):  
        offspring = breeding(population, fitness, min_values = min_values, max_values = max_values, mu = mu, elite = elite, size = size, target_function = target_function) 
        population = mutation(offspring, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, size = size, target_function = target_function)
        fitness = fitness_function(population)
        value = np.copy(population[population[:,-1].argsort()][0,:])
        if(elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value) 
        count = count + 1      
    return elite_ind 

###############################################################################

# Function: Distance Matrix
def distance_matrix(dataset, criteria = 0):
    distance_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for i in range(0, distance_array.shape[0]):
        for j in range(0, distance_array.shape[1]):
            distance_array[i,j] = dataset[i, criteria] - dataset[j, criteria] 
    return distance_array

# Function: Preferences
def preference_degree(dataset, W, Q, S, P, F):
    pd_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for k in range(0, dataset.shape[1]):
        distance_array = distance_matrix(dataset, criteria = k)
        for i in range(0, distance_array.shape[0]):
            for j in range(0, distance_array.shape[1]):
                if (i != j):
                    if (F[k] == 't1'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't2'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't3'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > 0 and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  = distance_array[i,j]/P[k]
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't4'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  = 0.5
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't5'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  =  (distance_array[i,j] - Q[k])/(P[k] -  Q[k])
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't6'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1 - math.exp(-(distance_array[i,j]**2)/(2*S[k]**2))
                    if (F[k] == 't7'):
                        if (distance_array[i,j] == 0):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > 0 and distance_array[i,j] <= S[k]):
                            distance_array[i,j]  =  (distance_array[i,j]/S[k])**0.5
                        elif (distance_array[i,j] > S[k] ):
                            distance_array[i,j] = 1
        pd_array = pd_array + W[k]*distance_array
    pd_array = pd_array/sum(W)
    return pd_array

# Function: Pre-Order Rank 
def po_ranking(po_string):
    alts   = list(range(1, po_string.shape[0] + 1)) 
    alts   = ['a' + str(alt) for alt in alts]
    for i in range (po_string.shape[0] - 1, -1, -1):
        for j in range (po_string.shape[1] -1, -1, -1):
            if (po_string[i,j] == 'I'):
                po_string = np.delete(po_string, i, axis = 0)
                po_string = np.delete(po_string, i, axis = 1)
                alts[j] = str(alts[j] + "; " + alts[i])
                del alts[i]
                break    
    graph = {}
    for i in range(po_string.shape[0]):
        if (len(alts[i]) == 0):
            graph[alts[i]] = i 
        else:
            graph[alts[i][ :2]] = i   
            graph[alts[i][-2:]] = i 
    po_matrix = np.zeros((po_string.shape[0], po_string.shape[1]))
    for i in range (0, po_string.shape[0]):
        for j in range (0, po_string.shape[1]):
            if (po_string[i,j] == 'P+'):
                po_matrix[i,j] = 1
    col_sum = np.sum(po_matrix, axis = 1)
    alts_rank = [x for _, x in sorted(zip(col_sum, alts))]
    if (np.sum(col_sum) != 0):
        alts_rank.reverse()      
    graph_rank = {}
    for i in range(po_string.shape[0]):
        if (len(alts_rank[i]) == 0):
            graph_rank[alts_rank[i]] = i 
        else:
            graph_rank[alts_rank[i][ :2]] = i   
            graph_rank[alts_rank[i][-2:]] = i
    rank = np.copy(po_matrix)
    for i in range(0, po_matrix.shape[0]):
        for j in range(0, po_matrix.shape[1]): 
            if (po_matrix[i,j] == 1):
                rank[i,:] = np.clip(rank[i,:] - rank[j,:], 0, 1)   
    rank_xy = np.zeros((len(alts_rank), 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        if (len(alts_rank) - np.sum(~rank.any(1)) != 0):
            rank_xy[i, 1] = len(alts_rank) - np.sum(~rank.any(1))
        else:
            rank_xy[i, 1] = 1
    for i in range(0, len(alts_rank) - 1):
        i1 = int(graph[alts_rank[ i ][:2]]) 
        i2 = int(graph[alts_rank[i+1][:2]])
        if (po_string[i1,i2] == 'P+'):
            rank_xy[i+1,1] = rank_xy[i+1,1] - 1
            for j in range(i+2, rank_xy.shape[0]):
                rank_xy[j,1] = rank_xy[i+1,1]
        if (po_string[i1,i2] == 'R'):
            rank_xy[i+1,0] = rank_xy[i,0] + 1            
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], alts_rank[i], size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, len(alts_rank)):
        alts_rank[i] = alts_rank[i][:2]
    for i in range(0, rank.shape[0]):
        for j in range(0, rank.shape[1]):
            k1 = int(graph_rank[list(graph.keys())[list(graph.values()).index(i)]])
            k2 = int(graph_rank[list(graph.keys())[list(graph.values()).index(j)]])
            if (rank[i, j] == 1):  
                plt.arrow(rank_xy[k1, 0], rank_xy[k1, 1], rank_xy[k2, 0] - rank_xy[k1, 0], rank_xy[k2, 1] - rank_xy[k1, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    xmin = np.amin(rank_xy[:,0])
    xmax = np.amax(rank_xy[:,0])
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

# Function: Pre-Order Rank 
def rank(po_string):
    alts   = list(range(1, po_string.shape[0] + 1)) 
    alts   = ['a' + str(alt) for alt in alts]  
    po_matrix = np.zeros((po_string.shape[0], po_string.shape[1]))
    for i in range (0, po_string.shape[0]):
        for j in range (0, po_string.shape[1]):
            if (po_string[i,j] == 'P+'):
                po_matrix[i,j] = 1
    col_sum = np.sum(po_matrix, axis = 1)
    alts_rank = [x for _, x in sorted(zip(col_sum, alts))]
    if (np.sum(col_sum) != 0):
        alts_rank.reverse()      
    alts_rank = [float(x.replace('a', ''))-1 for x in alts_rank]
    return alts_rank

###############################################################################

# Function: Promethee I
def promethee_i(dataset, W, Q, S, P, F, graph = False):
    pd_matrix  = preference_degree(dataset, W, Q, S, P, F)
    alts       = list(range(1, pd_matrix.shape[0] + 1)) 
    alts       = ['a' + str(alt) for alt in alts]
    flow_plus  = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
    flow_minus = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
    cp_matrix  = np.empty((pd_matrix.shape[0], pd_matrix.shape[0]), dtype = 'U25')
    cp_matrix.fill('-')
    for i in range(0, cp_matrix.shape[0]):
          for j in range(0, cp_matrix.shape[0]): 
              if ( (flow_plus[i] > flow_plus[j] and flow_minus[i] < flow_minus[j]) or (flow_plus[i] == flow_plus[j] and flow_minus[i] < flow_minus[j]) or (flow_plus[i] > flow_plus[j] and flow_minus[i] == flow_minus[j])):
                  cp_matrix[i,j] = 'P+'
              if (flow_plus[i] == flow_plus[j] and flow_minus[i] == flow_minus[j] and i != j):
                  cp_matrix[i,j] = 'I'    
              if ( (flow_plus[i] > flow_plus[j] and flow_minus[i] > flow_minus[j]) or (flow_plus[i] < flow_plus[j] and flow_minus[i] < flow_minus[j]) ):
                  cp_matrix[i,j] = 'R'
    if (graph == True):
       po_ranking(cp_matrix) 
    return cp_matrix

###############################################################################
