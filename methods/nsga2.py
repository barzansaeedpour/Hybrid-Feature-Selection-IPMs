

from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from parameters import Parameters
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from objective_functions import ObjectiveFunctions
from cost_function import CostFunction
from pymoo.core.mutation import Mutation
from tqdm import tqdm

def nsga2(dataset_model):

    parameters = Parameters()
    

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

    class MyMutation(Mutation):
        def _do(self, problem, X, **kwargs):
            
            nf = dataset_model.all_x.shape[1]
            for i in range(X.shape[0]):
                mutate = np.random.choice([0, 1], nf, p=[1-parameters.probabilty_of_mutation, parameters.probabilty_of_mutation])
                X[i] = np.abs(mutate-X[i])           
            return X
        
    class MyCallback(Callback):

        def __init__(self) -> None:
            super().__init__()
            t = tqdm(total=parameters.n_gen, ncols=80)
            self.t = t
            self.t.set_description(f'NSGA II in Progress for {dataset_model.dataset_name}')
            objectiveFunctions = ObjectiveFunctions()
            for o in objectiveFunctions.objectives:
                self.data[o["name"]] = []
                self.data[o["name"]+"_best"] = []
                self.data[o["name"]+"_best_solution"]=[]

        def notify(self, algorithm):
            self.t.update(1)
            f = algorithm.pop.get("F")
            objectiveFunctions = ObjectiveFunctions()
            for i, o in enumerate(objectiveFunctions.objectives):
                self.data[o["name"]].append(algorithm.pop.get("F")[:,i].mean())
                self.data[o["name"]+"_best"].append(algorithm.pop.get("F")[:,i].min())
                index = np.argmin(algorithm.pop.get("F")[:,i])
                self.data[o["name"]+"_best_solution"].append(algorithm.pop.get("X")[index])

    problem = MyProblem()


    algorithm = NSGA2(pop_size=parameters.pop_size,
                      sampling=MySampling(),
                      crossover=TwoPointCrossover(),
                      mutation=MyMutation(),
                      )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', parameters.n_gen),
                   seed=1,
                   save_history=False,
                   callback = MyCallback(),
                   verbose=False)

    return res
    
