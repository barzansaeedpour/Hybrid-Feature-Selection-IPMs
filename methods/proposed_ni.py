


from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
import pandas as pd
from parameters2 import Parameters
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from objective_functions import ObjectiveFunctions
from cost_function import CostFunction
import math
from mmlcr.mmlcr import MMLCR
import matplotlib.pyplot as plt
from static_variables import StaticVariables
import os
from pymoo.core.repair import Repair
from tqdm import tqdm


def exponential_model_function(hp, lp, d, power):
    power = power
    F = list(range(1,d+1))
    p=[]
    a = np.log(hp/lp) / (d**power-1)
    b = 1 / hp*((hp/lp)**(1/(d**power-1)))
    for f in F:
        p.append(1/(b*(math.e**(a*(f**power-2)))))
    
    # convex_model = p
    return p

class ProbabilityDistributions:
    convex_model =[]
    linear_model=[]
    concave_model=[]
    constant_model=[]

    def __init__(self,hp,lp,d) -> None:
        self.convex_model_function(hp, lp, d)
        self.linear_model_function(hp, lp, d)
        self.concave_model_function(hp, lp, d)
        self.constant_model_function(d)

    def convex_model_function(self, hp, lp, d):

        power = 0.1
        F = list(range(1,d+1))
        p=[]
        a = np.log(hp/lp) / (d**power-1)
        b = 1 / hp*((hp/lp)**(1/(d**power-1)))
        for f in F:
            p.append(1/(b*(math.e**(a*(f**power-2)))))
        ProbabilityDistributions.convex_model = p
        return p

    def linear_model_function(self,hp,lp,d):
        F = list(range(1,d+1))
        p=[]
        for f in F:
            p.append(((lp-hp)*(f-1) + d*hp - hp)/(d-1))

        ProbabilityDistributions.linear_model = p
        return p

    def concave_model_function(self, hp, lp, d):

        power = 6
        F = list(range(1,d+1))
        p=[]
        a = np.log(hp/lp) / (d**power-1)
        b = 1 / hp*((hp/lp)**(1/(d**power-1)))
        for f in F:
            p.append(1/(b*(math.e**(a*(f**power-2)))))
        ProbabilityDistributions.concave_model = p
        return p

    def constant_model_function(self, d):
        F = list(range(1,d+1))
        p=[]
        for f in F:
            p.append(1/2)
        ProbabilityDistributions.constant_model = p
        return p

def proposed_ni(dataset_model, model_name):

    parameters = Parameters()
    static_vars = StaticVariables()
    pop_size = parameters.pop_size
    nf = dataset_model.all_x.shape[1]

    result_base_directory = static_vars.base_directory + dataset_model.dataset_name +"/"+model_name+"/"
    # result_base_directory = static_vars.base_directory
    try:
        # os.makedirs(result_base_directory)
        os.makedirs(result_base_directory+'probability/')
    except:
        pass


    def repair(p):
        for i,f in enumerate(p):
            if f >0.9:
                p[i]= 0.9
            elif f<0.1:
                p[i]= 0.1
        return p

    def my_plot(p,d, save=False,title='', alpha=1 , xlabel= "filter ranks", label = 'not_defined', save_dir='', iteration=0):
        F = list(range(1,d+1))
        plt.ioff()
        plt.figure(save_dir)
        if title == '':
            plt.title(f"PMFI functions - {dataset_model.dataset_name} - iteration: {iteration}")
        else: 
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("probability of importance")
        # plt.xticks(list(range(0,len(F)+1)))
        plt.yticks(np.arange(0,1.1,0.1))
        if save:
            plt.plot(F, p, marker = "*" , label=label,alpha=alpha)
            plt.legend()
            plt.savefig(save_dir)
            plt.savefig(save_dir.replace('.png','.svg'),format="svg",dpi=1200,bbox_inches='tight',facecolor='white')
    


    mMlcr = MMLCR(dataset_model)
    feature_ranks, features_order_by_rank = mMlcr.get_sorted_features()


    # initialize probability distributions
    hp , lp = parameters.hp, parameters.lp
    d = dataset_model.all_x.shape[1]
    probabilityDistributions = ProbabilityDistributions(hp=hp,lp=lp,d=d)


    class MyProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=dataset_model.all_x.shape[1],
                n_obj=parameters.n_obj,
                xu=1,
                xl=0,
                vtype=int
            )

        def _evaluate(self, x, out, *args, **kwargs):
            costFunction = CostFunction(pop=x,dataset_model= dataset_model)
            temp =costFunction.evaluate()
            out["F"] = temp

    class MySampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            nf = dataset_model.all_x.shape[1] # number of features
            X = np.zeros([parameters.pop_size, nf],dtype=int)
            for i in range(X.shape[0]):
                individual = np.zeros(nf,dtype=int)
                for f in range(nf):
                    r2 = np.random.rand()
                    if (r2<0.5):
                        individual[f] = 1
                    else:
                        individual[f] = 0
                X[i] = individual
            return X

    class MyCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            t = tqdm(total=parameters.n_gen, ncols=80)
            self.t = t
            self.t.set_description(f'Proposed-ni in Progress for {dataset_model.dataset_name}')
            self.initial_lambda =parameters.initial_lambda
            self.final_lambda =parameters.final_lambda
            self.iteration_of_update =parameters.iteration_of_update
            n = (parameters.n_gen)/self.iteration_of_update
            try:
                self.d = (self.final_lambda - self.initial_lambda) / (n - 1)
            except:
                self.d = 0
            self.Lambda  = self.initial_lambda 
            
            objectiveFunctions = ObjectiveFunctions()
            for o in objectiveFunctions.objectives:
                self.data[o["name"]] = []
                self.data[o["name"]+"_best"] = []
                self.data[o["name"]+"_best_solution"]=[]

        def notify( self, algorithm ):
            self.t.update(1)

            X = algorithm.pop.get("X")
            p = np.sum(X,axis=0)/pop_size # frequency
            
            PD = ProbabilityDistributions
            
            if algorithm.n_iter % self.iteration_of_update == 0:
                
                if algorithm.n_iter == self.iteration_of_update:
                    self.Lambda = self.initial_lambda # in the first update, Lambda is initial_lambda
                else:
                    self.Lambda = self.Lambda + self.d # in the other updates, we also update the lambda
                
                for f in range(nf):
                    if (p[features_order_by_rank[f]]) >= PD.convex_model[f]:
                        PD.convex_model[f] = (PD.convex_model[f]) + ( self.Lambda * np.abs(p[features_order_by_rank[f]]-PD.convex_model[f]))
                    else:
                        PD.convex_model[f] = (PD.convex_model[f]) - ( self.Lambda * np.abs(p[features_order_by_rank[f]]-PD.convex_model[f]) )
                        
                    if (p[features_order_by_rank[f]]) >= PD.linear_model[f]:
                        PD.linear_model[f] = (PD.linear_model[f]) + (self.Lambda * np.abs(p[features_order_by_rank[f]]-PD.linear_model[f]))
                    else:
                        PD.linear_model[f] = (PD.linear_model[f]) - (self.Lambda * np.abs(p[features_order_by_rank[f]]-PD.linear_model[f]))

                    if (p[features_order_by_rank[f]]) >= PD.concave_model[f]:
                        PD.concave_model[f] = (PD.concave_model[f]) + (self.Lambda * np.abs(p[features_order_by_rank[f]]-PD.concave_model[f]) )
                    else:
                        PD.concave_model[f] = (PD.concave_model[f]) - (self.Lambda * np.abs(p[features_order_by_rank[f]]-PD.concave_model[f]) )
                    
                    PD.convex_model = repair(PD.convex_model)
                    PD.linear_model = repair(PD.linear_model)
                    PD.concave_model = repair(PD.concave_model)
                   
            # sort p
            pp = np.zeros(nf) 
            for f in range(nf):
                pp[f] = p[features_order_by_rank[f]]
            my_plot(pp,nf, save=True ,label='wrapper_result', save_dir = result_base_directory+'probability/' + 'common_'+str(algorithm.n_iter)+'.png', iteration=algorithm.n_iter)
            my_plot(PD.convex_model,nf, save=True ,label='convex_model', save_dir = result_base_directory+'probability/' + 'common_'+str(algorithm.n_iter)+'.png', iteration=algorithm.n_iter)
            my_plot(PD.linear_model,nf, save=True ,label='linear_model', save_dir = result_base_directory+'probability/' + 'common_'+str(algorithm.n_iter)+'.png', iteration=algorithm.n_iter)
            my_plot(PD.concave_model,nf, save=True ,label='concave_model', save_dir = result_base_directory+'probability/' + 'common_'+str(algorithm.n_iter)+'.png', iteration=algorithm.n_iter)
                
            if algorithm.n_iter==parameters.n_gen: 
                '''
                in the last iteration, calculate the mean of PMFI and sort it as 
                the final fearue ranking of the proposed algorithm.
                '''
                s = np.add(PD.convex_model ,PD.linear_model )
                s = np.add(s,PD.concave_model)
                mean = np.divide(s,3)
                mean = list(mean)
                

                final_feature_ranking = list(np.argsort(mean)[::-1])
                R = np.array(final_feature_ranking)
                df = pd.DataFrame(R)
                df.to_csv(f"{result_base_directory}/final_feature_ranking.csv")


                mean.sort(reverse=True)

                save_dir = result_base_directory+'probability/' + 'final_feature_ranking'+ '.png'
                my_plot(pp,nf, save=True ,alpha = 0.3, label='wrapper_result', save_dir = save_dir, iteration=algorithm.n_iter)
                my_plot(PD.convex_model,nf, save=True , alpha = 0.3, label='convex_model', save_dir = save_dir, iteration=algorithm.n_iter)
                my_plot(PD.linear_model,nf, save=True ,alpha = 0.3, label='linear_model', save_dir = save_dir, iteration=algorithm.n_iter)
                my_plot(PD.concave_model,nf, save=True ,alpha = 0.3, label='concave_model', save_dir = save_dir, iteration=algorithm.n_iter)
                
                my_plot(mean,nf,title=f"Sorted final probabilities - {dataset_model.dataset_name}", save=True ,xlabel = 'feature' ,label='final feature ranking', save_dir = result_base_directory+'probability/' + 'final_feature_ranking'+ '.png', iteration=algorithm.n_iter)
            
            
            f = algorithm.pop.get("F")
            objectiveFunctions = ObjectiveFunctions()
            for i, o in enumerate(objectiveFunctions.objectives):
                self.data[o["name"]].append(algorithm.pop.get("F")[:,i].mean())
                self.data[o["name"]+"_best"].append(algorithm.pop.get("F")[:,i].min())
                index = np.argmin(algorithm.pop.get("F")[:,i])
                self.data[o["name"]+"_best_solution"].append(algorithm.pop.get("X")[index])
    
    class MyMutation(Mutation):
        def _do(self, problem, X, **kwargs):
            PD = ProbabilityDistributions
            for i in range(X.shape[0]):
                mutate = np.random.choice([0, 1], nf, p=[1-parameters.probability_distribution_mutation_percentage, parameters.probability_distribution_mutation_percentage])
                if True: #  local search filter
                    r1 = np.random.rand()
                    if r1 < 0.20:
                        p = PD.convex_model
                    elif r1 < 0.40:
                        p = PD.concave_model
                    elif r1 < 0.60:
                        p = PD.linear_model
                    else:
                        p = PD.constant_model
                    for j, m in enumerate(mutate):
                        if m==0 :
                            X[i,j] = X[i,j] # no mutation
                        else:
                            feature_rank = feature_ranks[j]
                            r2 = np.random.rand()
                            if(r2 < p[feature_rank]):
                                X[i,j] = 1 
                            else:
                                X[i,j] = 0             
            return X

    class RepairUnvalidSolutions(Repair):
        def _do(self, problem, Z, **kwargs):
            for i in range(len(Z)):
                z = Z[i]
                if len(np.nonzero(z)[0])==0:
                    z = np.random.randint(0,2,nf, dtype= int)
                Z[i] = z
            return Z

    problem = MyProblem()


    algorithm = NSGA2(pop_size=parameters.pop_size,
                      crossover=TwoPointCrossover(),
                    mutation=MyMutation(),
                    sampling=MySampling(),
                    repair=RepairUnvalidSolutions()
                      )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', parameters.n_gen),
                   seed=1,
                   save_history=False,
                   callback = MyCallback(),
                   verbose=False)

    return res
    
