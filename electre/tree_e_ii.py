###############################################################################

import copy
import math
import numpy as np
import random
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from electre.util_e_ii import electre_ii, genetic_algorithm

###############################################################################

# Main    
def tree_electre_ii(dataset, target_assignment = [], W = [], c_minus = [], c_zero = [], c_plus = [], d_minus = [], d_plus = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.70, number_of_models = 100):
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
        weights_t     = variable_list[0:random_dataset.shape[1]]
        c_minus_t     = variable_list[-5]
        c_zero_t      = variable_list[-4]
        c_plus_t      = variable_list[-3]
        d_minus_t     = variable_list[-2]
        d_plus_t      = variable_list[-1]
        _, _, _, _, _, _, e_ii, _ = electre_ii(dataset = random_dataset, W = weights_t, c_minus = c_minus_t, c_zero = c_zero_t, c_plus = c_plus_t, d_minus = d_minus_t, d_plus = d_plus_t, graph = False)
        e_ii = e_ii[:,0]
        e_ii = e_ii.tolist()
        kendall_tau, _ = stats.kendalltau(random_y, e_ii)
        if (math.isnan(kendall_tau)):
            kendall_tau = -1
        return -kendall_tau
    
    while count < number_of_models:
        random_dataset = np.copy(dataset)
        random_y       = np.copy(y)
        random_W       = copy.deepcopy(W)
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
        if (isinstance(c_minus, list)):
            min_values.extend([0]) #c_minus
            max_values.extend([1]) #c_minus
        else:
            min_values.extend([c_minus]) #c_minus
            max_values.extend([c_minus]) #c_minus
        if (isinstance(c_zero, list)):
            min_values.extend([0]) #c_zero
            max_values.extend([1]) #c_zero
        else:
            min_values.extend([c_zero]) #c_zero
            max_values.extend([c_zero]) #c_zero
        if (isinstance(c_plus, list)):
            min_values.extend([0]) #c_plus
            max_values.extend([1]) #c_plus
        else:
            min_values.extend([c_plus]) #c_plus
            max_values.extend([c_plus]) #c_plus
        if (isinstance(d_minus, list)):
            min_values.extend([0]) #d_minus
            max_values.extend([1]) #d_minus
        else:
            min_values.extend([d_minus]) #d_minus
            max_values.extend([d_minus]) #d_minus
        if (isinstance(d_plus, list)):
            min_values.extend([0]) #d_plus
            max_values.extend([1]) #d_plus
        else:
            min_values.extend([d_plus]) #d_plus
            max_values.extend([d_plus]) #d_plus
        ga = genetic_algorithm(population_size = population_size, mutation_rate = mutation_rate, elite = elite, min_values = min_values, max_values = max_values, eta = eta, mu = mu, generations = generations, target_function = target_function)
        weights_ga  = ga[0:random_dataset.shape[1]]
        c_minus_ga  = ga[-6]
        c_zero_ga   = ga[-5]
        c_plus_ga   = ga[-4]
        d_minus_ga  = ga[-3]
        d_plus_ga   = ga[-2]
        kendall_tau = ga[-1]*(-1)
        _, _, _, _, _, _, y_hat, _ = electre_ii(dataset = random_dataset, W = weights_ga, c_minus = c_minus_ga, c_zero = c_zero_ga, c_plus = c_plus_ga, d_minus = d_minus_ga, d_plus = d_plus_ga, graph = False)
        y_hat = y_hat[:,0]
        y_hat = y_hat.tolist()
        ensemble_model.append([weights_ga, kendall_tau, criteria, criteria_remove, cases_remove, y_hat, random_y, c_minus_ga, c_zero_ga, c_plus_ga, d_minus_ga, d_plus_ga])
        count = count + 1
        print('Model # ' + str(count) ) 
    return ensemble_model

###############################################################################
    
# Prediction
def predict_e_ii(models, dataset, verbose = True):
    prediction     = []
    solutions      = [[]]*dataset.shape[0]
    ensemble_model = copy.deepcopy(models)
    for i in range(0, len(ensemble_model)):
        alternatives = np.copy(dataset)  
        alternatives = np.delete(alternatives, ensemble_model[i][3], axis = 1)
        _, _, _, _, _, _, rank, _ = electre_ii(dataset = alternatives, W = ensemble_model[i][0], c_minus = ensemble_model[i][7], c_zero = ensemble_model[i][8], c_plus = ensemble_model[i][9], d_minus = ensemble_model[i][10], d_plus = ensemble_model[i][11], graph = False)
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
def metrics_e_ii(models):
    ensemble_model      = copy.deepcopy(models)   
    features_importance = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    count_features      = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    mean_features       = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    std_features        = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    kdl_mean            = 0
    kdl_std             = 0
    cm_mean             = 0
    cm_std              = 0
    cz_mean             = 0
    cz_std              = 0
    cp_mean             = 0
    cp_std              = 0
    dm_mean             = 0
    dm_std              = 0
    dp_mean             = 0
    dp_std              = 0
    for i in range(0, len(ensemble_model)):
        weights  = ensemble_model[i][0]   
        criteria = ensemble_model[i][2]  
        kdl_mean = kdl_mean + ensemble_model[i][1]
        cm_mean  = cm_mean  + ensemble_model[i][7]
        cz_mean  = cz_mean  + ensemble_model[i][8]
        cp_mean  = cp_mean  + ensemble_model[i][9]
        dm_mean  = dm_mean  + ensemble_model[i][10]
        dp_mean  = dp_mean  + ensemble_model[i][11]
        for j in range(0, len(criteria)):
            features_importance[criteria[j]] = features_importance[criteria[j]] + weights[j] 
            count_features[criteria[j]]      = count_features[criteria[j]] + 1
    kdl_mean = kdl_mean/len(ensemble_model)
    cm_mean  = cm_mean/len(ensemble_model)
    cz_mean  = cz_mean/len(ensemble_model)
    cp_mean  = cp_mean/len(ensemble_model)
    dm_mean  = dm_mean/len(ensemble_model)
    dp_mean  = dp_mean/len(ensemble_model)
    for i in range(0, len(mean_features)):
        mean_features[i]    = features_importance[i]/count_features[i]
    for i in range(0, len(ensemble_model)):  
        weights  = ensemble_model[i][0] 
        criteria = ensemble_model[i][2] 
        kdl_std  = kdl_std + (ensemble_model[i][1] - kdl_mean)**2
        cm_std   = cm_std  + (ensemble_model[i][1] -  cm_mean)**2
        cz_std   = cz_std  + (ensemble_model[i][1] -  cz_mean)**2
        cp_std   = cp_std  + (ensemble_model[i][1] -  cp_mean)**2
        dm_std   = dm_std  + (ensemble_model[i][1] -  dm_mean)**2
        dp_std   = dp_std  + (ensemble_model[i][1] -  dp_mean)**2
        for j in range(0, len(criteria)):
            std_features[criteria[j]]    = std_features[criteria[j]]    + (weights[j] - mean_features[criteria[j]])**2
    kdl_std  = (kdl_std/(len(ensemble_model)-1))**(1/2)
    cm_std   = (cm_std /(len(ensemble_model)-1))**(1/2)
    cz_std   = (cz_std /(len(ensemble_model)-1))**(1/2)
    cp_std   = (cp_std /(len(ensemble_model)-1))**(1/2)
    dm_std   = (dm_std /(len(ensemble_model)-1))**(1/2)
    dp_std   = (dp_std /(len(ensemble_model)-1))**(1/2)
    for i in range(0, len(std_features)): 
         std_features[i]    = (std_features[i]/(count_features[i]-1))**(1/2)
    return mean_features, std_features, kdl_mean, kdl_std, cm_mean, cm_std, cz_mean, cz_std, cp_mean, cp_std, dm_mean, dm_std, dp_mean, dp_std 

###############################################################################
