import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

AIRFOILS_PATH = "dataset/airfoils_dataset/"
NBODY_PATH = "dataset/nbody_dataset/"
pos = "current_wp"
current_wp=os.getcwd()
if pos == "snap":
    EXP_PATH = "/dfs/project/plasma/results/"
elif pos == "current_wp":
    EXP_PATH =current_wp+"/results/"
else:
    EXP_PATH = "./results/"