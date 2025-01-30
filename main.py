
import warnings
warnings.filterwarnings("ignore")
from load_methods import Method
from multi_label_datasets.load_dataset import load_my_dataset
from static_variables import StaticVariables
import os
from objective_functions import ObjectiveFunctions
from plot_settings import PlotSettings
from plot_results import plot_results
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs('results',exist_ok=True)

# initializing the static variables
static_vars = StaticVariables()


# inputs: dataset model, method model
# outputs: saved results

method_names = ["nsga2","proposed-ni","proposed-nm","proposed"]
dataset_names = ["medical"]
 
method_models=[]
for method_name in method_names:
    method_models.append(Method(name = method_name))

dataset_models=[]
for dataset_name in dataset_names:
    dataset_models.append(load_my_dataset(dataset_name=dataset_name))

for dataset_model in dataset_models:
    max_value = {}
    min_value = {}
    objectiveFunctions = ObjectiveFunctions()
    for o in objectiveFunctions.objectives:
        if o["type"] == "maximization":
            max_value[o["name"]] = 0
        if o["type"] == "minimization":
            min_value[o["name"]] = 9999
     
    for method_model in method_models:
        plot_results(method_model, dataset_model.dataset_name, max_value, min_value)







 





