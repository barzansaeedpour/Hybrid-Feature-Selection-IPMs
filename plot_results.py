
from static_variables import StaticVariables
import os
import numpy as np
import pandas as pd
from objective_functions import ObjectiveFunctions
import matplotlib.pyplot as plt
from plot_settings import PlotSettings
from numpy import random

import numpy as np
import math
import matplotlib.pyplot as plt


def plot_results(method_model, dataset_name, max_value, min_value):
   
    static_vars = StaticVariables()
    load_directory = static_vars.base_directory + dataset_name +"/"+method_model.name+"/"
    
    objectiveFunctions = ObjectiveFunctions()
    
    for o in objectiveFunctions.objectives:

        df = pd.read_csv(load_directory + o["name"]+"_for_plot"+".csv")
        data = np.array(df)[:,1] # drop the index
        plt.figure(o["name"]+"_" + dataset_name )
        plt.title(o["abbreviation"]+" - " + dataset_name )
        plt.ylabel(o["plot_name"])
        plt.xlabel("iteration")
        plt.ioff() 
        plt.plot(data,marker= method_model.marker, c= method_model.color, label= method_model.plotName)
        
        plt.legend()
        plt.savefig(load_directory.replace(f"/{method_model.name}","")+o["name"]+'.png')
        plt.savefig(load_directory.replace(f"/{method_model.name}","")+o["name"]+'.svg',format="svg",dpi=1200,bbox_inches='tight',facecolor='white')
       

