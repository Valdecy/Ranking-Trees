###############################################################################

import copy
import math
import numpy as np
import random
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from electre.util_e_iv import electre_iv, genetic_algorithm

###############################################################################

# Main    
def tree_electre_iv(dataset, target_assignment = [], P = [], Q = [], V = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.70, number_of_models = 100):
    count           = 0
    ensemble_model  = []
    if (len(target_assignment) == 0):
        tSVD      = TruncatedSVD(n_components = 1, n_iter = 100, random_state = 42)
        tSVD_proj = tSVD.fit_transform(dataset)
        y         = np.copy(tSVD_proj)
        flat      = tSVD_proj.flatten()
        flat.sort()
        count_1 = 0
        for i in range(y.shape[0] - 1, -1, -1):
            for j in range(0, y.shape[0]):
                if (y[j] == flat[i]):
                    y[j] = count_1
                    if (flat[i] not in y):
                        count_1 = count_1 + 1
    elif (len(target_assignment) > 0):
        y = target_assignment
        
    def target_function(variable_list):
        variable_list = variable_list.tolist()
        Q_t = variable_list[0:random_dataset.shape[1]]
        P_t = variable_list[random_dataset.shape[1]*1:random_dataset.shape[1]*2]
        V_t = variable_list[random_dataset.shape[1]*2:random_dataset.shape[1]*3]
        _, _, _, e_iv, _ = electre_iv(dataset = random_dataset, Q = Q_t, P = P_t, V = V_t, graph = False)
        e_iv = e_iv[:,0]
        e_iv = e_iv.tolist()
        kendall_tau, _ = stats.kendalltau(random_y, e_iv)
        if (math.isnan(kendall_tau)):
            kendall_tau = -1
        return -kendall_tau
    
    while count < number_of_models:
        random_dataset = np.copy(dataset)
        random_y       = np.copy(y)
        random_Q       = copy.deepcopy(Q)
        random_P       = copy.deepcopy(P)
        random_V       = copy.deepcopy(V)
        if (random_dataset.shape[1] > 2):
            criteria_remove = random.sample(list(range(0, dataset.shape[1])), random.randint(1, dataset.shape[1]- 2))
            random_dataset  = np.delete(random_dataset, criteria_remove, axis = 1)
        else:
            criteria_remove = []
        criteria_remove.sort(reverse = True)
        for i in range(dataset.shape[1] - 1, -1, -1):
            if i in criteria_remove:
                if (len(random_Q) > 0):
                    del random_Q[i]
                if (len(random_P) > 0):
                    del random_P[i]
                if (len(random_V) > 0):
                    del random_V[i]
        criteria     =  [item for item in list(range(0, dataset.shape[1])) if item not in criteria_remove]
        cases_remove = random.sample(list(range(0, dataset.shape[0])), math.ceil(dataset.shape[0]*(1 - samples)))             
        if (len(cases_remove) > 0):
            random_dataset  = np.delete(random_dataset, cases_remove, axis = 0)
            random_y        = np.delete(random_y, cases_remove, axis = 0)
            random_y        = random_y[:,0]
            random_y        = random_y.tolist()
        if (len(random_Q) == 0):  
            min_values = [0.00]*random_dataset.shape[1]
            max_values = list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0))
        elif (len(random_Q) > 0):
            min_values = copy.deepcopy(random_Q)
            max_values = copy.deepcopy(random_Q)
        if (len(random_P) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend(list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0)))
        elif (len(random_P) > 0):
            min_values.extend(copy.deepcopy(random_P))
            max_values.extend(copy.deepcopy(random_P))
        if (len(random_V) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend(list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0)))
        elif (len(random_V) > 0):
            min_values.extend(copy.deepcopy(random_V))
            max_values.extend(copy.deepcopy(random_V))
        ga = genetic_algorithm(population_size = population_size, mutation_rate = mutation_rate, elite = elite, min_values = min_values, max_values = max_values, eta = eta, mu = mu, generations = generations, size = random_dataset.shape[1], target_function = target_function)
        Q_ga = ga[0:random_dataset.shape[1]]
        P_ga = ga[random_dataset.shape[1]*1:random_dataset.shape[1]*2]
        V_ga = ga[random_dataset.shape[1]*2:random_dataset.shape[1]*3]
        kendall_tau = ga[-1]*(-1)
        _, _, _, y_hat, _ = electre_iv(dataset = random_dataset, Q = Q_ga, P = P_ga, V = V_ga, graph = False)
        y_hat = y_hat[:,0]
        y_hat = y_hat.tolist()
        ensemble_model.append([kendall_tau, criteria, criteria_remove, cases_remove, y_hat, random_y, Q_ga, P_ga, V_ga])
        count = count + 1
        print('Model # ' + str(count) ) 
    return ensemble_model

###############################################################################
    
# Prediction
def predict_e_iv(models, dataset, verbose = True):
    prediction     = []
    solutions      = [[]]*dataset.shape[0]
    ensemble_model = copy.deepcopy(models)
    for i in range(0, len(ensemble_model)):
        alternatives = np.copy(dataset)  
        alternatives = np.delete(alternatives, ensemble_model[i][2], axis = 1)
        _, _, _, rank, _ = electre_iv(dataset = alternatives, Q = ensemble_model[i][6], P = ensemble_model[i][7], V = ensemble_model[i][8], graph = False)
        rank = rank[:,0]
        rank = rank.tolist()
        for j in range(0, len(solutions)):
            if (i == 0):
                solutions[j] = [rank[j]]
            else:
                solutions[j].extend([rank[j]])
    for i in range(0, dataset.shape[0]):
       prediction.append(int(max(set(solutions[i]), key = solutions[i].count)))
       if (verbose == True):
           print('a' + str(i + 1) + ' = ' + str(prediction[i]))
    return prediction, solutions

###############################################################################

# Metrics
def metrics_e_iv(models):
    ensemble_model      = copy.deepcopy(models)   
    count_features      = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    kdl_mean            = 0
    kdl_std             = 0
    q_tresholds         = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    q_tresholds_mean    = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    q_tresholds_std     = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    p_tresholds         = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    p_tresholds_mean    = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    p_tresholds_std     = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    v_tresholds         = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    v_tresholds_mean    = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    v_tresholds_std     = [0]*(len(ensemble_model[0][1]) + len(ensemble_model[0][2]))
    for i in range(0, len(ensemble_model)):  
        criteria = ensemble_model[i][1] 
        kdl_mean = kdl_mean + ensemble_model[i][0]
        q        = ensemble_model[i][6] 
        p        = ensemble_model[i][7] 
        v        = ensemble_model[i][8] 
        for j in range(0, len(criteria)):
            q_tresholds[criteria[j]]         = q_tresholds[criteria[j]] + q[j]
            p_tresholds[criteria[j]]         = p_tresholds[criteria[j]] + p[j]
            v_tresholds[criteria[j]]         = v_tresholds[criteria[j]] + v[j]
            count_features[criteria[j]]      = count_features[criteria[j]] + 1
    kdl_mean = kdl_mean/len(ensemble_model)
    for i in range(0, len(q_tresholds_mean)):
        q_tresholds_mean[i] = q_tresholds[i]/count_features[i]
        p_tresholds_mean[i] = p_tresholds[i]/count_features[i]
        v_tresholds_mean[i] = v_tresholds[i]/count_features[i]
    for i in range(0, len(ensemble_model)):  
        criteria = ensemble_model[i][1] 
        kdl_std  = kdl_std + (ensemble_model[i][0] - kdl_mean)**2
        q        = ensemble_model[i][6] 
        p        = ensemble_model[i][7] 
        v        = ensemble_model[i][8] 
        for j in range(0, len(criteria)):
            q_tresholds_std[criteria[j]] = q_tresholds_std[criteria[j]] + (q[j] - q_tresholds_mean[criteria[j]])**2
            p_tresholds_std[criteria[j]] = p_tresholds_std[criteria[j]] + (p[j] - p_tresholds_mean[criteria[j]])**2
            v_tresholds_std[criteria[j]] = v_tresholds_std[criteria[j]] + (v[j] - v_tresholds_mean[criteria[j]])**2
    kdl_std  = (kdl_std/(len(ensemble_model)-1))**(1/2)
    for i in range(0, len(q_tresholds_std)): 
         q_tresholds_std[i] = (q_tresholds_std[i]/(count_features[i]-1))**(1/2)
         p_tresholds_std[i] = (p_tresholds_std[i]/(count_features[i]-1))**(1/2)
         v_tresholds_std[i] = (v_tresholds_std[i]/(count_features[i]-1))**(1/2)
    return kdl_mean, kdl_std, q_tresholds_mean, q_tresholds_std, p_tresholds_mean, p_tresholds_std, v_tresholds_mean, v_tresholds_std

###############################################################################
