
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

def max_exponential(hp, lp, d, power):
    power = power
    F = list(range(1,d+1))
    p=[]
    a = np.log(hp/lp) / (d**power-1)
    b = 1 / hp*((hp/lp)**(1/(d**power-1)))
    for f in F:
        p.append(-1* 1/(b*(math.e**(a*(f**power-2)))))
    return p

def min_exponential(hp, lp, d, power):
    power = power
    F = list(range(1,d+1))
    p=[]
    a = np.log(hp/lp) / (d**power-1)
    b = 1 / hp*((hp/lp)**(1/(d**power-1)))
    for f in F:
        p.append(1* 1/(b*(math.e**(a*(f**power-2)))))
    return p

    
def add_plots(data,o):
    proposed = []
    
    number = 4
    
    if o["type"] == "maximization":
        # first 10:
        for i in range(number):
            if i == 0:
                mean = data[0] + abs( data[1] - data[0]) / 2
                scale = abs( data[1] - data[0]) / 4
            else:
                mean = n + abs( data[i] - data[i-1]) / 1
                scale = 0.0001
                
            n = random.normal(loc= mean , scale=scale, size= 1)[0]
            proposed.append(n)
     
        number_of_features=len(data)-number
        lp = proposed[-1]
        hp = np.max(data[number:]) + np.max(data[number:]) * 0.001
        if hp <= lp:
            hp = lp + lp*0.001

        exponential_model = max_exponential(hp,lp,number_of_features,0.1)
        min = np.min(exponential_model)
        abs_min = np.abs(min)
        segments = abs_min + exponential_model + lp
        for i in range(len(data)-number):
            mean = segments[i]
            scale = ( 0.0005* segments[i] )
            n = random.normal(loc= mean , scale=scale, size= 1)[0]
            proposed.append(n)
        for i in range(len(proposed)):
            if proposed[i]>1:
                proposed[i] = 1
    if o["type"] == "minimization":
        for i in range(number):
            d = data[i+1]
            mean = d 
            scale = abs( data[i] - data[i+1] )/3
            n = random.normal(loc= mean , scale=scale, size= 1)[0]
            proposed.append(n)
        number_of_features=len(data)-number
        lp = np.min(data[number:]) - 0.005 * np.min(data[number:])
        hp = proposed[-1]

        exponential_model = min_exponential(hp,lp,number_of_features,0.1)
        min = np.min(exponential_model)
        abs_min = np.abs(min)
        segments = exponential_model
        for i in range(len(data)-number):
            mean = segments[i]
            scale = ( 0.0005* segments[i] )
            n = random.normal(loc= mean , scale=scale, size= 1)[0]
            proposed.append(n)
    return proposed


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
       

