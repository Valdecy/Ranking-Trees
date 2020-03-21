###############################################################################

# Required Libraries
import copy
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Rank  Plot
def rank_plot(pred): 
    prediction = copy.deepcopy(pred)
    alts   = list(range(1, len(prediction) + 1)) 
    alts   = ['a' + str(alt) for alt in alts]
    for i in range (len(prediction) - 1, -1, -1):
        for j in range (len(prediction) -1, -1, -1):
            if (i != j and prediction[i] == prediction[j]):
                alts[j] = str(alts[j] + "; " + alts[i])
                del alts[i]
                del prediction[i]
                break  
    prediction, alts = zip(*sorted(zip(prediction, alts)))
    rank_xy = np.zeros((len(prediction), 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = len(prediction)-i           
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], alts[i], size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))           
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
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

###############################################################################