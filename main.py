###############################################################################

# Importing Libraries
import numpy as np

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
models_e_iii = tree_electre_iii(dataset, target_assignment = [], W = [], P = [], Q = [], V = [], elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.05, generations = 150, samples = 0.70, number_of_models = 100)

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
