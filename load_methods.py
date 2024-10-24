
from methods.nsga2 import nsga2
from methods.proposed_ni import proposed_ni
from methods.proposed_nm import proposed_nm
from methods.proposed import proposed
from save_results import save_results



class Method:
    def __init__(self, name):
        self.name = name
        self.plotName = name
        self.color = 'black' # the default color for plots
        self.marker = 'o'    # the default marker for plots

        # options:
        # 'cyan','purple', 'black', 'blue', 'orange', 'lightgray', 'lime','seagreen','navy','slateblue','teal','olive','navajowhite','tan','red','yellow'
        # '.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

        if self.name == "nsga2":
            self.color = 'purple'
            self.marker = '^'
            self.plotName = 'NSGA-II'
            
            
        elif self.name == "proposed-ni":
            self.color = 'orange'
            self.marker = 'o'
            self.plotName = 'Proposed-ni'
        
        elif self.name == "proposed-nm":
            self.color = 'blue'
            self.marker = 's'
            self.plotName = 'Proposed-nm'
            
        elif self.name == "proposed":
            self.color = 'cyan'
            self.marker = '*'
            self.plotName = 'Proposed'

            
    def feature_selection(self, dataset_model):
        # print(f"{self.name} selects the best features for {dataset_model.dataset_name}")
        
        if self.name == "nsga2":
            res = nsga2(dataset_model=dataset_model)
            save_results(res= res,dataset_name=dataset_model.dataset_name, method_model=self)
        
        elif self.name == "proposed-ni":
            res = proposed_ni(dataset_model=dataset_model, model_name= self.name)
            save_results(res= res,dataset_name=dataset_model.dataset_name, method_model=self)
        
        elif self.name == "proposed-nm":
            res = proposed_nm(dataset_model=dataset_model, model_name= self.name)
            save_results(res= res,dataset_name=dataset_model.dataset_name, method_model=self)
        
        elif self.name == "proposed":
            res = proposed(dataset_model=dataset_model, model_name= self.name)
            save_results(res= res,dataset_name=dataset_model.dataset_name, method_model=self)




    
    