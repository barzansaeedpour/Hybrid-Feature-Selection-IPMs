import math


class ObjectiveFunctions:

    def __init__(self,):
        self.objectives = [
            {
                "name":"hamming_loss",
                "plot_name":"Hamming Loss",
                "abbreviation":'HL (\u2193)',
                "type": "minimization",
                "best_solution":[],
                "best_value":math.inf,
            },
            {
                "name":"macro_f1",
                "plot_name":"Macro-F1",
                "abbreviation":'MaF1 (\u2191)',
                "type": "maximization",
                "best_solution":[],
                "best_value":math.inf,
            },
            {
                "name":"micro_f1",
                "plot_name":"Micro-F1",
                "abbreviation":'MiF1 (\u2191)',
                "type": "maximization",
                "best_solution":[],
                "best_value":math.inf,
            },
            {
                "name":"ranking_loss",
                "plot_name":"Ranking Loss",
                "abbreviation": 'RL (\u2193)',
                "type": "minimization",
                "best_solution":[],
                "best_value":math.inf,
            },
            {
                "name":"average_precision",
                "plot_name":"Average Precision",
                "abbreviation":'AP (\u2191)',
                "type": "maximization",
                "best_solution":[],
                "best_value":math.inf,
            },
            {
                "name":"coverage_error",
                "plot_name":"Coverage Error",
                "abbreviation": 'CV (\u2193)',
                "type": "minimization",
                "best_solution":[],
                "best_value":math.inf,
            },
        ]

    
