import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
from sklearn.preprocessing import KBinsDiscretizer
import sys
import os
import time
sys.path.append("./")
# import my_module
# top_package = __import__(__name__.split('.')[0])
from multi_label_datasets.load_dataset import load_my_dataset
import random 
import numpy as np
import scipy as sp
from tqdm import tqdm
from pyitlib import discrete_random_variable as drv

def mi(X, Y, message = None): 
    mi = [[None for y in range(Y.shape[1])] for x in range(X.shape[1])]
    if X.shape[1] > 1:
        with tqdm(total=X.shape[1], ncols=80) as t:
            t.set_description('Calculating {}'.format(message))
            for x in range(X.shape[1]):
                for y in range(Y.shape[1]): 
                        mi[x][y] = drv.information_mutual(np.transpose(X[:,x].reshape(-1,1)), \
                            np.transpose(Y[:,y].reshape(-1,1)))
                t.update(1)
    else: 
        for x in range(X.shape[1]):
            for y in range(Y.shape[1]): 
                    mi[x][y] = drv.information_mutual(np.transpose(X[:,x].reshape(-1,1)), \
                        np.transpose(Y[:,y].reshape(-1,1)))
    return mi

class MMLCR:
    def __init__(self, datasetModel):
        self.x = datasetModel.all_x
        self.y = datasetModel.all_y
        self.datasetName = datasetModel.dataset_name
        self.dimension = self.x.shape[1]
        self.nSamples = self.x.shape[0]
        self.nf = self.x.shape[1]
        self.nl = self.y.shape[1]

        feature_label_correlation_matrix = np.zeros([self.nf,self.nl])
        # feature_feature_correlation_matrix = np.zeros([self.nf,self.nf])

        save_directory= './saved/correlation/'
        try:
            os.makedirs(save_directory)
        except:
            pass
        save_directory= save_directory+self.datasetName

        try:
            # x = 1/0
            feature_label_correlation_matrix= np.loadtxt(save_directory+'_flcorr.txt')
        except:
            # for i in range(self.nf):
            #     for j in range(self.nl):
            #         temp = pearsonr(self.x[:,i],self.y[:,j])[0]
            #         if math.isnan(temp):
            #             temp = 0
            #         feature_label_correlation_matrix[i,j] = temp
                    
            # np.savetxt(save_directory+'_flcorr.txt', feature_label_correlation_matrix)
            est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
            est.fit(self.x)
            X = est.transform(self.x).astype(int)
            Y = self.y.astype(int)
            feature_label_correlation_matrix = np.array(mi(X,Y,message="feature-label mutual information"))
            feature_label_correlation_matrix = feature_label_correlation_matrix.reshape(self.nf,self.nl)
            np.savetxt(save_directory+'_flcorr.txt', feature_label_correlation_matrix)
            
            # feature_feature_correlation_matrix = np.array(mi(X,X))
            # feature_feature_correlation_matrix = feature_feature_correlation_matrix.reshape(self.nf,self.nf)
            # np.savetxt(save_directory+'_ffcorr.txt', feature_feature_correlation_matrix)
            
            
            # for i in range(self.nf):
            #     for j in range(i,self.nf):
            #         temp = pearsonr(self.x[:,i],self.x[:,j])[0]
            #         if math.isnan(temp):
            #             temp = 0
            #         feature_feature_correlation_matrix[i,j] = temp
            
            # for i in range(self.nf):
            #     for j in range(i+1):
            #         feature_feature_correlation_matrix[i,j] = feature_feature_correlation_matrix[j,i]

            # np.savetxt(save_directory+'_ffcorr.txt', feature_feature_correlation_matrix)

        self.feature_label_correlation_matrix = feature_label_correlation_matrix
        # self.feature_feature_correlation_matrix = feature_feature_correlation_matrix

        # feature_label_correlation_matrix = np.sqrt(np.sum(feature_label_correlation_matrix**2,axis =1))
        # feature_feature_correlation_matrix = np.sqrt(np.sum(feature_feature_correlation_matrix**2,axis =1))

        
    # def entropy(self,vec, base=2):
    #     " Returns the empirical entropy H(X) in the input vector."
    #     _, vec = np.unique(vec, return_counts=True)
    #     prob_vec = np.array(vec/float(sum(vec)))
    #     if base == 2:
    #         logfn = np.log2
    #     elif base == 10:
    #         logfn = np.log10
    #     else:
    #         logfn = np.log
    #     return prob_vec.dot(-logfn(prob_vec))

    # def conditional_entropy(self,x, y):
    #     "Returns H(X|Y)."
    #     uy, uyc = np.unique(y, return_counts=True)
    #     prob_uyc = uyc/float(sum(uyc))
    #     cond_entropy_x = np.array([self.entropy(x[y == v]) for v in uy])
    #     return prob_uyc.dot(cond_entropy_x)
        
    # def mutual_information(self,x, y):
    #     " Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
    #     return self.entropy(x) - self.conditional_entropy(x, y)

    # def symmetrical_uncertainty(self,x, y):
    #     " Returns 'symmetrical uncertainty' - a symmetric mutual information measure."
    #     return 2.0* self.mutual_information(x, y)/(self.entropy(x) + self.entropy(y))
        

    class FeatureModel():
        def __init__(self, feature_number=0, value = [], fl_corr=[], cluster_number=0, rank=0, d=0):
            self.feature_number = feature_number
            self.value = np.array(value) 
            self.fl_corr = np.array(fl_corr)
            self.cluster_number = cluster_number
            self.rank = rank
            self.d = d

        def toString(self):
            print("feature_number: ",self.feature_number)
            print("value: ",self.value)
            print("fl_corr: ",self.fl_corr)
            print("cluster_number: ",self.cluster_number)
            print("rank: ",self.rank)
            print("d: ",self.d)
            print("*****************************")
            print("\n")

    class LabelModel():
        def __init__(self, value = np.array([])):
            self.value = value
            

    def get_sorted_features(self):
        x = self.x
        y = self.y
        datasetName =self.datasetName
        dimension = self.dimension
        nSamples = self.nSamples
        nf = self.nf
        nl = self.nl
        
        all_features = np.array([])
        for i in range(nf):
            all_features=np.append(all_features,self.FeatureModel(feature_number=i,value=x[:,i]))
   
        
        for i in range(nf):
            for j in range(nl):
                all_features[i].fl_corr = np.append(all_features[i].fl_corr , self.feature_label_correlation_matrix[i,j])

        FLC = np.zeros([nf,nl]) # Feature-Label Correlation Matrix
        # symmetrical_uncertainty = fcbf(x,y)
        for i in range(nf):
            for j in range(nl):
                f = x[:,i]
                l = y[:,j]        
                FLC[i,j] = self.feature_label_correlation_matrix[i,j]  
        
        
        z = np.zeros((1,nl))
        
        def distance(v1, v2):
            return np.sqrt(np.sum((v1 - v2) ** 2))

        FFD = np.zeros([nf,nf]) # Feature-Feature Distance Matrix
        for i in range(nf):
            for j in range(nf):
                FFD[i,j]= distance(FLC[i],FLC[j]) 
                
        # number of clusters will be 20 percent.
        number_of_clusters = round(0.2 * nf)
        # print(number_of_clusters)
        
        kmn = KMeans(n_clusters = number_of_clusters) 
        kmn.fit(FFD)
        labels = kmn.predict(FFD)
        
    
        d=[]
        for i in range(nf):
            d.append(distance(z,FLC[i])) 
            all_features[i].d = distance(z,FLC[i])
        
        for i in range(nf):
            all_features[i].cluster_number = labels[i]
            
        for i in range(number_of_clusters):
            clusters = [f for f in all_features if f.cluster_number == i]
            # for c in clusters:
            #     c.toString()
        
        all_features_sorte_by_d = np.array([])
        for i in range(number_of_clusters):
            clusters = [f for f in all_features if f.cluster_number == i]
      
            newlist = sorted(clusters, key=lambda x: x.d,reverse=True)
            rank = 1
            
            for c in newlist:
                c.rank= rank
                rank = rank+1
                # c.toString()
                all_features_sorte_by_d = np.append(all_features_sorte_by_d , c)
        
        all_rankes = [f.rank for f in all_features_sorte_by_d]
        max_ranked = max(all_rankes)

        w=np.array([])
        for i in range(max_ranked):
            ranked = [f for f in all_features_sorte_by_d if f.rank == i+1] # find all features with rank i+1
            sorted_ranked = sorted(ranked, key=lambda x: x.d,reverse=True) # sort them based on d (descending)  
            w = np.append(w,sorted_ranked)

        features_order_by_rank = [ n.feature_number for n in w]
        
        feature_ranks= [] 
        for i in range(nf):
            for j in range(nf):
                if w[j].feature_number == i:
                    feature_ranks.append(j)
         
        # features_order_by_rank=[60,87,88,57,55,71,48,96,52,50,78,66,49,77,89,102,97,56,46,91,80,25,10,79,86,100,45,94,81,99,92,93,85,95,82,14,15,2,19,64,18,83,44,22,13,6,33,61,68,32,3,72,53,26,17,63,51,29,23,69,73,84,8,75,31,34,4,43,0,41,20,11,30,1,70,62,90,39,28,16,47,74,24,76,40,36,21,12,58,67,65,101,54,98,37,59,27,5,35,38,7,9,42]
        
        # for i,f in enumerate(features_order_by_rank):
        #     feature_ranks[f]=i
        
        return  feature_ranks, features_order_by_rank


# datasets = [
#      { "name":'cal500' ,
#       "nf":68,
#      },
#      { "name":'emotions',
#       "nf":72,
#      },
#      { "name":'enron',
#       "nf":1001,
#      },
#      { "name":'flags' ,
#       "nf":19,
#      },
#      { "name":'genbase',
#       "nf":1186,
#      },
#      { "name":'gnegative',
#       "nf":1717,
#      },
#      { "name":'gpositive',
#       "nf":912,
#      },
#      { "name":'image',
#       "nf":294,
#      },
#      { "name":'medical',
#       "nf":1449,
#      },
#      { "name":'scene',
#       "nf":294,
#      },
#      { "name":'virus',
#       "nf":749,
#      },
#      { "name":'yeast',
#       "nf":103,
#      },
#     ]

# datasets = [m['name'] for m in datasets]

# # datasets = ['emotions']
# for d in datasets:
#     dataset_model = load_my_dataset(dataset_name=d)
#     start = time.time()
#     mMlcr = MMLCR(dataset_model)
#     feature_ranks, features_order_by_rank = mMlcr.get_sorted_features()
#     end = time.time()
    
    
#     filename = f'./PyIT_MLFS/output/RunningTimes/{dataset_model.dataset_name}/'
#     os.makedirs(filename, exist_ok=True)
#     filename = filename + 'MMLCR.txt'
#     np.savetxt(filename, [end-start], fmt = '%d')

#     filename = f'./PyIT_MLFS/output/SelectedSubsets/{dataset_model.dataset_name}/'
#     os.makedirs(filename, exist_ok=True)
#     filename = filename + 'MMLCR.csv'
#     np.savetxt(filename, features_order_by_rank, delimiter=',', fmt = '%d')


