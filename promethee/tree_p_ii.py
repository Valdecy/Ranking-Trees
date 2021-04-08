###############################################################################

import copy
import math
import numpy as np
import random
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import TruncatedSVD
from promethee.util_p_ii import promethee_ii, genetic_algorithm

###############################################################################

# Main    
def tree_promethee_ii(dataset, target_assignment = [], W = [], Q = [], P = [], S = [], F = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.70, number_of_models = 100):
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
        W_t = variable_list[0:random_dataset.shape[1]]
        Q_t = variable_list[random_dataset.shape[1]*1:random_dataset.shape[1]*2]
        P_t = variable_list[random_dataset.shape[1]*2:random_dataset.shape[1]*3]
        S_t = variable_list[random_dataset.shape[1]*3:random_dataset.shape[1]*4]
        F_t = variable_list[random_dataset.shape[1]*4:random_dataset.shape[1]*5]
        p_ii = promethee_ii(dataset = random_dataset, W = W_t, Q = Q_t, S = S_t, P = P_t, F = F_t)
        p_ii = p_ii[:,0]
        p_ii = p_ii.tolist()
        kendall_tau, _ = stats.kendalltau(random_y, p_ii)
        if (math.isnan(kendall_tau)):
            kendall_tau = -1
        return -kendall_tau
    
    while count < number_of_models:
        random_dataset = np.copy(dataset)
        random_y       = np.copy(y)
        random_W       = copy.deepcopy(W)
        random_Q       = copy.deepcopy(Q)
        random_P       = copy.deepcopy(P)
        random_S       = copy.deepcopy(S)
        random_F       = copy.deepcopy(tranform_shape(F, strg = True))
        if (random_dataset.shape[1] > 2):
            criteria_remove = random.sample(list(range(0, dataset.shape[1])), random.randint(1, dataset.shape[1]- 2))
            random_dataset  = np.delete(random_dataset, criteria_remove, axis = 1)
        else:
            criteria_remove = []
        criteria_remove.sort(reverse = True)
        for i in range(dataset.shape[1] - 1, -1, -1):
            if i in criteria_remove:
                if (len(random_W) > 0):
                    del random_W[i]
                if (len(random_Q) > 0):
                    del random_Q[i]
                if (len(random_P) > 0):
                    del random_P[i]
                if (len(random_S) > 0):
                    del random_S[i]
                if (len(random_F) > 0):
                    del random_F[i]
        criteria     =  [item for item in list(range(0, dataset.shape[1])) if item not in criteria_remove]
        cases_remove = random.sample(list(range(0, dataset.shape[0])), math.ceil(dataset.shape[0]*(1 - samples)))             
        if (len(cases_remove) > 0):
            random_dataset  = np.delete(random_dataset, cases_remove, axis = 0)
            random_y        = np.delete(random_y, cases_remove, axis = 0)
            random_y        = random_y[:,0]
            random_y        = random_y.tolist()
        if (len(random_W) == 0):  
            min_values = [0.00]*random_dataset.shape[1]
            max_values = [1.00]*random_dataset.shape[1]
        elif (len(random_W) > 0):
            min_values = copy.deepcopy(random_W)
            max_values = copy.deepcopy(random_W)
        if (len(random_Q) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend(list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0)))
        elif (len(random_Q) > 0):
            min_values.extend(copy.deepcopy(random_Q))
            max_values.extend(copy.deepcopy(random_Q))
        if (len(random_P) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend(list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0)))
        elif (len(random_P) > 0):
            min_values.extend(copy.deepcopy(random_P))
            max_values.extend(copy.deepcopy(random_P))
        if (len(random_S) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend([1.00]*random_dataset.shape[1])
        elif (len(random_S) > 0):
            min_values.extend(copy.deepcopy(random_S))
            max_values.extend(copy.deepcopy(random_S))
        if (len(random_F) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend([1.00]*random_dataset.shape[1])
        elif (len(random_F) > 0):
            min_values.extend(copy.deepcopy(random_F))
            max_values.extend(copy.deepcopy(random_F))
        ga = genetic_algorithm(population_size = population_size, mutation_rate = mutation_rate, elite = elite, min_values = min_values, max_values = max_values, eta = eta, mu = mu, generations = generations, size = random_dataset.shape[1], target_function = target_function)
        W_ga = ga[0:random_dataset.shape[1]]
        Q_ga = ga[random_dataset.shape[1]*1:random_dataset.shape[1]*2]
        P_ga = ga[random_dataset.shape[1]*2:random_dataset.shape[1]*3]
        S_ga = ga[random_dataset.shape[1]*3:random_dataset.shape[1]*4]
        F_ga = ga[random_dataset.shape[1]*4:random_dataset.shape[1]*5]
        F_ga = tranform_shape(F_ga, strg = False)
        for i in range(0, len(S_ga)):
            if (F_ga[i] != 't7'):
                S_ga[i] = 0
        kendall_tau = ga[-1]*(-1)
        y_hat = promethee_ii(dataset = random_dataset, W = W_ga, Q = Q_ga, S = S_ga, P = P_ga, F = F_ga)
        y_hat = y_hat[:,0]
        y_hat = y_hat.tolist()
        ensemble_model.append([W_ga, kendall_tau, criteria, criteria_remove, cases_remove, y_hat, random_y, Q_ga, P_ga, S_ga, F_ga])
        count = count + 1
        print('Model # ' + str(count) ) 
    return ensemble_model

###############################################################################
    
# Prediction
def predict_p_ii(models, dataset, verbose = True):
    prediction     = []
    solutions      = [[]]*dataset.shape[0]
    ensemble_model = copy.deepcopy(models)
    for i in range(0, len(ensemble_model)):
        alternatives = np.copy(dataset)  
        alternatives = np.delete(alternatives, ensemble_model[i][3], axis = 1)
        rank = promethee_ii(dataset = alternatives, W = ensemble_model[i][0], Q = ensemble_model[i][7], S = ensemble_model[i][9], P = ensemble_model[i][8], F = ensemble_model[i][10])
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

# Shape Dictionary
def tranform_shape(F, strg = True):
    transformed = []
    if (strg == True):
        for i in range(0, len(F)):
            if (F[i] == 't1'):
                transformed.append(0/7)
            if (F[i] == 't2'):
                transformed.append(1/7)
            if (F[i] == 't3'):
                transformed.append(2/7)
            if (F[i] == 't4'):
                transformed.append(3/7)
            if (F[i] == 't5'):
                transformed.append(4/7)
            if (F[i] == 't6'):
                transformed.append(5/7)
            if (F[i] == 't7'):
                transformed.append(6/7)
    else:
        for i in range(0, len(F)):
            if (F[i] < 1/7):
                transformed.append('t1')
            if (F[i] >= 1/7 and F[i] < 2/7):
                transformed.append('t2')
            if (F[i] >= 2/7 and F[i] < 3/7):
                transformed.append('t3')
            if (F[i] >= 3/7 and F[i] < 4/7):
                transformed.append('t4')
            if (F[i] >= 4/7 and F[i] < 5/7):
                transformed.append('t5')
            if (F[i] >= 5/7 and F[i] < 6/7):
                transformed.append('t6')
            if (F[i] >= 6/7 ):
                transformed.append('t7')
    return transformed

###############################################################################

# Metrics
def metrics_p_ii(models):
    ensemble_model      = copy.deepcopy(models)   
    features_importance = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    count_features      = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    mean_features       = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    std_features        = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    kdl_mean            = 0
    kdl_std             = 0
    q_tresholds         = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    q_tresholds_mean    = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    q_tresholds_std     = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    p_tresholds         = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    p_tresholds_mean    = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    p_tresholds_std     = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    s_tresholds         = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    s_tresholds_mean    = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    s_tresholds_std     = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    f_tresholds         = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    f_tresholds_mean    = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    for i in range(0, len(ensemble_model)):
        weights  = ensemble_model[i][0]   
        criteria = ensemble_model[i][2] 
        kdl_mean = kdl_mean + ensemble_model[i][1]
        q        = ensemble_model[i][7] 
        p        = ensemble_model[i][8] 
        s        = ensemble_model[i][9] 
        f        = tranform_shape(ensemble_model[i][10], strg = True)
        for j in range(0, len(criteria)):
            features_importance[criteria[j]] = features_importance[criteria[j]] + weights[j] 
            q_tresholds[criteria[j]]         = q_tresholds[criteria[j]] + q[j]
            p_tresholds[criteria[j]]         = p_tresholds[criteria[j]] + p[j]
            s_tresholds[criteria[j]]         = s_tresholds[criteria[j]] + s[j]
            f_tresholds[criteria[j]]         = f_tresholds[criteria[j]] + f[j]
            count_features[criteria[j]]      = count_features[criteria[j]] + 1
    kdl_mean = kdl_mean/len(ensemble_model)
    for i in range(0, len(mean_features)):
        mean_features[i]    = features_importance[i]/count_features[i]
        q_tresholds_mean[i] = q_tresholds[i]/count_features[i]
        p_tresholds_mean[i] = p_tresholds[i]/count_features[i]
        s_tresholds_mean[i] = s_tresholds[i]/count_features[i]
        f_tresholds_mean[i] = s_tresholds[i]/count_features[i]
    for i in range(0, len(ensemble_model)):  
        weights  = ensemble_model[i][0] 
        criteria = ensemble_model[i][2] 
        kdl_std  = kdl_std + (ensemble_model[i][1] - kdl_mean)**2
        q        = ensemble_model[i][7] 
        p        = ensemble_model[i][8] 
        s        = ensemble_model[i][9] 
        for j in range(0, len(criteria)):
            std_features[criteria[j]]    = std_features[criteria[j]]    + (weights[j] - mean_features[criteria[j]])**2
            q_tresholds_std[criteria[j]] = q_tresholds_std[criteria[j]] + (q[j] - q_tresholds_mean[criteria[j]])**2
            p_tresholds_std[criteria[j]] = p_tresholds_std[criteria[j]] + (p[j] - p_tresholds_mean[criteria[j]])**2
            s_tresholds_std[criteria[j]] = s_tresholds_std[criteria[j]] + (s[j] - s_tresholds_mean[criteria[j]])**2
    kdl_std  = (kdl_std/(len(ensemble_model)-1))**(1/2)
    for i in range(0, len(std_features)): 
         std_features[i]    = (std_features[i]/(count_features[i]-1))**(1/2)
         q_tresholds_std[i] = (q_tresholds_std[i]/(count_features[i]-1))**(1/2)
         p_tresholds_std[i] = (p_tresholds_std[i]/(count_features[i]-1))**(1/2)
         s_tresholds_std[i] = (s_tresholds_std[i]/(count_features[i]-1))**(1/2)
    f_tresholds_mean = tranform_shape(f_tresholds_mean, strg = False)
    return mean_features, std_features, kdl_mean, kdl_std, q_tresholds_mean, q_tresholds_std, p_tresholds_mean, p_tresholds_std, s_tresholds_mean, s_tresholds_std, f_tresholds_mean

###############################################################################
