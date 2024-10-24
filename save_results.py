
from static_variables import StaticVariables
import os
import numpy as np
import pandas as pd
from objective_functions import ObjectiveFunctions
import matplotlib.pyplot as plt
from plot_settings import PlotSettings
from pymoo.indicators.hv import HV
from hypervolume import hypervolume

def save_results(res,method_model,dataset_name):
    static_vars = StaticVariables()
    save_directory = static_vars.base_directory + dataset_name +"/"+method_model.name+"/"
    try:
        os.makedirs(save_directory)
    except:
        pass
    
    pop = res.pop # the final population
    

    F = res.F

    ref_point = np.array([1, 0, 0, 1, 0 , 10]) # worse values
    ind = HV(ref_point=ref_point)

    pareto_front_hv = []
    for f in F:
        # print("HV", ind(f))
        pareto_front_hv.append(ind(f))

    hv = np.array(pareto_front_hv)
    df = pd.DataFrame(hv)
    df.to_csv(f"{save_directory}/hv.csv")
    
    # with open(f"{save_directory}/HV.txt",'w') as f:
    #     f.write(f"mean hv: {sum(pareto_front_hv) / len(pareto_front_hv)}")

    hv = hypervolume(rp = ref_point, pf=F)

    for k,v in {"X":res.X , "F":res.F}.items():
        R = np.array(v)
        df = pd.DataFrame(R)
        df.to_csv(f"{save_directory}/{k}.csv")
    
    callback_data = res.algorithm.callback.data

    for k,v in callback_data.items():
        R = np.array(v)
        df = pd.DataFrame(R)
        df.to_csv(f"{save_directory}/{k}.csv")
    

    objectiveFunctions = ObjectiveFunctions()
    
    for o in objectiveFunctions.objectives:
        # data = callback_data[o["name"]+"_best"]
        data = callback_data[o["name"]]
        if o["type"] == "maximization": 
            data  = np.multiply(-1,data)

        R = np.array(data)
        df = pd.DataFrame(R)
        # df.to_csv(save_directory.replace(f"/{method_model.name}","")+o["name"]+".csv")
        df.to_csv(save_directory + o["name"]+"_for_plot"+".csv")
        
        # plt.figure(o["name"]+"_" + dataset_name )
        # plt.title(o["name"]+"_" + dataset_name )
        # plt.ylabel(o["name"])
        # plt.xlabel("iteration")
        # plt.ioff()
        # # p = PlotSettings
        # # plt.plot(data,marker= p.markers[p.current_marker_index], c= p.colors[p.current_marker_index], label= method_model.name)
        # plt.plot(data,marker= method_model.marker, c= method_model.color, label= method_model.name)
        # plt.legend()
        # plt.savefig(save_directory.replace(f"/{method_model.name}","")+o["name"]+'.png')
        # plt.savefig(save_directory.replace(f"/{method_model.name}","")+o["name"]+'.svg',format="svg",dpi=1200,bbox_inches='tight',facecolor='white')
       
    # for k,v in callback_data.items():
    #     if 'best'in k:
    #         R = np.array(v)
    #         plt.figure( k + dataset_name +model_name)
    #         plt.plot(R)
    #         plt.title(k)
    #         plt.savefig(save_directory+k+'.png')
       