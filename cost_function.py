import numpy as np
from parameters import Parameters
from skmultilearn.adapt import MLkNN 
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

class CostFunction:
    def __init__(self, pop, dataset_model) -> None:
        self.pop = pop
        self.dataset_model = dataset_model
    
    def evaluate(self):
        costs = []
        X = self.pop
        x = self.dataset_model.all_x
        y = self.dataset_model.all_y
        nSamples = x.shape[0]
        parameters= Parameters()

        for i in range(X.shape[0]):
            individual=X[i]
            selected_features = np.nonzero(individual)
            number_of_selected_features = len(selected_features[0])
            new_data_set = x[:,selected_features].reshape(nSamples,number_of_selected_features)

            ranking_loss_list=[]
            average_precision_list=[]
            coverage_error_list=[]
            hamming_loss_list=[]
            accuracy_list=[]
            f1_score_list=[]
            macro_f1_list=[]
            micro_f1_list=[]
            # parameters = Parameters()
            for k in range(parameters.cost_function_evaluation_time):
                X_train, X_test, y_train, y_test = train_test_split(new_data_set, y, shuffle=parameters.shuffle, random_state = parameters.random_state, test_size=0.20)
                classifier = MLkNN(k=10)
                classifier.fit(X_train, y_train)
                prediction = classifier.predict(X_test).toarray()
                scores = classifier.predict_proba(X_test).toarray()
                ranking_loss_list.append( metrics.label_ranking_loss(y_test, scores))
                average_precision_list.append( metrics.average_precision_score(y_test, scores,average="samples"))
                coverage_error_list.append( metrics.coverage_error(y_test, scores))
                coverage_error_list.append( metrics.coverage_error(y_test, scores))
                hamming_loss_list.append( metrics.hamming_loss(y_test, prediction))
                accuracy_list.append( metrics.accuracy_score(y_test, prediction, normalize=True, sample_weight=None))
                f1_score_list.append( metrics.f1_score(y_true=y_test, y_pred=prediction, average='samples'))
                macro_f1_list.append( metrics.f1_score(y_true=y_test, y_pred=prediction, average='macro'))
                micro_f1_list.append( metrics.f1_score(y_true=y_test, y_pred=prediction, average='micro'))
            
            res = [min(hamming_loss_list),-max(macro_f1_list),-max(micro_f1_list),min(ranking_loss_list),-max(average_precision_list),min(coverage_error_list)]
            
            costs.append(res)
        return np.array(costs)