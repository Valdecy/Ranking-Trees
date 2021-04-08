###############################################################################

# Importing Libraries
import numpy as np
import time

from electre.tree_e_ii  import tree_electre_ii
from electre.tree_e_ii  import predict_e_ii
from electre.tree_e_ii  import metrics_e_ii
from electre.tree_e_iii import tree_electre_iii
from electre.tree_e_iii import predict_e_iii
from electre.tree_e_iii import metrics_e_iii
from electre.tree_e_iv  import tree_electre_iv
from electre.tree_e_iv  import predict_e_iv
from electre.tree_e_iv  import metrics_e_iv

from electre.util       import rank_plot
from electre.util_e_ii  import electre_ii
from electre.util_e_iii import electre_iii
from electre.util_e_iv  import electre_iv

from promethee.tree_p_i   import tree_promethee_i
from promethee.tree_p_i   import predict_p_i
from promethee.tree_p_i   import metrics_p_i
from promethee.tree_p_ii  import tree_promethee_ii
from promethee.tree_p_ii  import predict_p_ii
from promethee.tree_p_ii  import metrics_p_ii
from promethee.tree_p_iii import tree_promethee_iii
from promethee.tree_p_iii import predict_p_iii
from promethee.tree_p_iii import metrics_p_iii
from promethee.tree_p_iv  import tree_promethee_iv
from promethee.tree_p_iv  import predict_p_iv
from promethee.tree_p_iv  import metrics_p_iv

from promethee.util_p_i   import promethee_i, rank
from promethee.util_p_ii  import promethee_ii
from promethee.util_p_iii import promethee_iii
from promethee.util_p_iv  import promethee_iv

##############################################################################

# loading Dataset
dataset = np.array([
                [1/589176 , 37188, 1/0.476, 0.00],   #a1
                [1/1548354, 45481, 1/0.600, 0.33],   #a2
                [1/2053485, 56623, 1/0.443, 0.00],   #a3
                [1/804270 , 29131, 1/0.474, 0.33],   #a4
                [1/2191952, 11177, 1/0.478, 0.67],   #a5
                [1/5181246, 10995, 1/0.500, 1.00],   #a6
                [1/2135702, 21794, 1/0.600, 0.00],   #a7
                [1/1547073, 21919, 1/0.496, 0.00]    #a8
                ])

dataset = dataset / dataset.max(axis = 0)

##############################################################################

# Electre II Example

# Build Models
models_e_ii = tree_electre_ii(dataset, target_assignment = [], W = [], c_minus = [], c_zero = [], c_plus = [], d_minus = [], d_plus = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.05, generations = 150, samples = 0.70, number_of_models = 100)

# Elicitated Parameters
e_ii_mean_features, e_ii_std_features, e_ii_kdl_mean, e_ii_kdl_std, e_ii_cm_mean, e_ii_cm_std, e_ii_cz_mean, e_ii_cz_std, e_ii_cp_mean, e_ii_cp_std, e_ii_dm_mean, e_ii_dm_std, e_ii_dp_mean, e_ii_dp_std = metrics_e_ii(models_e_ii)

# Rank with Elicitated Parameters
concordance_e_ii, discordance_e_ii, dominance_s_e_ii, dominance_w_e_ii, rank_D_e_ii, rank_A_e_ii, rank_M_e_ii, rank_P_e_ii = electre_ii(dataset, W = e_ii_mean_features, c_minus = e_ii_cm_mean, c_zero = e_ii_cz_mean, c_plus = e_ii_cp_mean, d_minus = e_ii_dm_mean, d_plus = e_ii_dp_mean, graph = True)

# Ensemble Rank Model
prediction_e_ii, solutions_e_ii = predict_e_ii(models_e_ii, dataset, verbose = True)

# Plot Ensemble Rank Model
rank_plot(prediction_e_ii)

##############################################################################

# Electre III Example

# Build Models
start_time = time.time()
models_e_iii = tree_electre_iii(dataset, target_assignment = [], W = [], P = [], Q = [], V = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.05, generations = 15, samples = 0.70, number_of_models = 100)
print("--- %s seconds ---" % (time.time() - start_time)) # 1534.4475753307343

# Elicitated Parameters
e_iii_mean_features, e_iii_std_features, e_iii_kdl_mean, e_iii_kdl_std, e_iii_q_tresholds_mean, e_iii_q_tresholds_std, e_iii_p_tresholds_mean, e_iii_p_tresholds_std, e_iii_v_tresholds_mean, e_iii_v_tresholds_std = metrics_e_iii(models_e_iii)

# Rank with Elicitated Parameters
global_concordance, credibility, rank_D, rank_A, rank_M, rank_P = electre_iii(dataset, P = e_iii_p_tresholds_mean, Q = e_iii_q_tresholds_mean, V = e_iii_v_tresholds_mean, W = e_iii_mean_features, graph = True)

# Ensemble Rank Model
prediction_iii, solutions_iii = predict_e_iii(models_e_iii, dataset, verbose = True)
rank_plot(prediction_iii)

##############################################################################

# Electre IV Example

# Build Models
models_e_iv = tree_electre_iv(dataset, target_assignment = [], P = [], Q = [], V = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.05, generations = 150, samples = 0.70, number_of_models = 100)

# Elicitated Parameters
e_iv_kdl_mean, e_iv_kdl_std, e_iv_q_tresholds_mean, e_iv_q_tresholds_std, e_iv_p_tresholds_mean, e_iv_p_tresholds_std, e_iv_v_tresholds_mean, e_iv_v_tresholds_std = metrics_e_iv(models_e_iv)

# Rank with Elicitated Parameters
credibility, rank_D, rank_A, rank_M, rank_P = electre_iv(dataset, P = e_iv_p_tresholds_mean, Q = e_iv_q_tresholds_mean, V = e_iv_v_tresholds_mean, graph = True)

# Ensemble Rank Model
prediction_e_iv, solutions_e_iv = predict_e_iv(models_e_iv, dataset, verbose = True)

# Plot Ensemble Rank Model
rank_plot(prediction_e_iv)

##############################################################################

# Promethee I Example

# Build Models
models_p_i = tree_promethee_i(dataset, target_assignment = [], W = [], Q = [], P = [], S = [], F = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.70, number_of_models = 10)

# Elicitated Parameters
p_i_mean_features, p_i_std_features, p_i_kdl_mean, p_i_kdl_std, p_i_q_tresholds_mean, p_i_q_tresholds_std, p_i_p_tresholds_mean, p_i_p_tresholds_std, p_i_s_tresholds_mean, p_i_s_tresholds_std, p_i_f_tresholds_mean = metrics_p_i(models_p_i)

# Rank with Elicitated Parameters
p_i = promethee_i(dataset = dataset, W = p_i_mean_features, Q = p_i_q_tresholds_mean, S = p_i_s_tresholds_mean, P = p_i_p_tresholds_mean, F = p_i_f_tresholds_mean)
rank_plot(rank(p_i))

# Ensemble Rank Model
prediction_p_i, solutions_p_i = predict_p_i(models_p_i, dataset, verbose = True)

# Plot Ensemble Rank Model
rank_plot(prediction_p_i)

##############################################################################

# Promethee II Example

# Build Models
models_p_ii = tree_promethee_ii(dataset, target_assignment = [], W = [], Q = [], P = [], S = [], F = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.70, number_of_models = 10)

# Elicitated Parameters
p_ii_mean_features, p_ii_std_features, p_ii_kdl_mean, p_ii_kdl_std, p_ii_q_tresholds_mean, p_ii_q_tresholds_std, p_ii_p_tresholds_mean, p_ii_p_tresholds_std, p_ii_s_tresholds_mean, p_ii_s_tresholds_std, p_ii_f_tresholds_mean = metrics_p_ii(models_p_ii)

# Rank with Elicitated Parameters
p_ii = promethee_ii(dataset = dataset, W = p_ii_mean_features, Q = p_ii_q_tresholds_mean, S = p_ii_s_tresholds_mean, P = p_ii_p_tresholds_mean, F = p_ii_f_tresholds_mean)

# Ensemble Rank Model
prediction_p_ii, solutions_p_ii = predict_p_ii(models_p_ii, dataset, verbose = True)

# Plot Ensemble Rank Model
rank_plot(prediction_p_ii)

##############################################################################

# Promethee III Example

# Build Models
models_p_iii = tree_promethee_iii(dataset, target_assignment = [], W = [], Q = [], P = [], S = [], F = [], lmbd = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.70, number_of_models = 10)

# Elicitated Parameters
p_iii_mean_features, p_iii_std_features, p_iii_kdl_mean, p_iii_kdl_std, p_iii_q_tresholds_mean, p_iii_q_tresholds_std, p_iii_p_tresholds_mean, p_iii_p_tresholds_std, p_iii_s_tresholds_mean, p_iii_s_tresholds_std, p_iii_f_tresholds_mean, lmbd_mean = metrics_p_iii(models_p_iii)

# Rank with Elicitated Parameters
p_iii = promethee_iii(dataset = dataset, W = p_iii_mean_features, Q = p_iii_q_tresholds_mean, S = p_iii_s_tresholds_mean, P = p_iii_p_tresholds_mean, F = p_iii_f_tresholds_mean, lmbd = lmbd_mean)
rank_plot(rank(p_iii))

# Ensemble Rank Model
prediction_p_iii, solutions_p_iii = predict_p_iii(models_p_iii, dataset, verbose = True)

# Plot Ensemble Rank Model
rank_plot(prediction_p_iii)

##############################################################################

# Promethee IV Example

# Build Models
models_p_iv = tree_promethee_iv(dataset, target_assignment = [], W = [], Q = [], P = [], S = [], F = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.70, number_of_models = 10)

# Elicitated Parameters
p_iv_mean_features, p_iv_std_features, p_iv_kdl_mean, p_iv_kdl_std, p_iv_q_tresholds_mean, p_iv_q_tresholds_std, p_iv_p_tresholds_mean, p_iv_p_tresholds_std, p_iv_s_tresholds_mean, p_iv_s_tresholds_std, p_iv_f_tresholds_mean = metrics_p_iv(models_p_iv)

# Rank with Elicitated Parameters
p_iv = promethee_iv(dataset = dataset, W = p_iv_mean_features, Q = p_iv_q_tresholds_mean, S = p_iv_s_tresholds_mean, P = p_iv_p_tresholds_mean, F = p_iv_f_tresholds_mean)

# Ensemble Rank Model
prediction_p_iv, solutions_p_iv = predict_p_iv(models_p_iv, dataset, verbose = True)

# Plot Ensemble Rank Model
rank_plot(prediction_p_iv)

##############################################################################
