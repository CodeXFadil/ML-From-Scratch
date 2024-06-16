## KNN

# Steps
import numpy as np
# define the Class with required inintializations

def Euclideon_Dist(a,b):
    distance = np.sqrt(sum((a-b)^2))
    return distance


class KNN_Clustering:
    
    def __init__(self, nclust, tol=0.001,maxiters = 100 ):
        self.nclust = nclust
        self.tol = tol 
        self.maxiters = maxiters 
    
    # CLuster Func
    def fit(self,X,nclust):
        
        self.X = X

        for iter in range(self.maxiters):
            
            Centroid = {}
            
            # Assigned n centroids based on nclust from the input data        
            for k in range(nclust):
                Centroid[k] = X[np.random.randint(1,len(X)-1)]
                    
            # Calualte the distance of each point from the Initized Centroid
            # Assigning the min duistance Centroid to the datapoints 
            
            # Initializing the Centroid data Dictionary
            Centroid_wise_data = {}
            for j in Centroid:
                Centroid_wise_data[j] = []

            for i in range(len(X)):
                distances = [Euclideon_Dist(Centroid[j],X[i]) for j in Centroid]
                Cluster_Centroid = distances.index(min(distances))
                Centroid_wise_data[Cluster_Centroid].append(X[i])
                
            # Calculate the New Centroid for each of the Grouped DataPoints/clusters 
            New_Centroid = {}
            for i in Centroid_wise_data:
                New_Centroid[i] = np.average(Centroid_wise_data[i],axis = 0)
                
            # Check to continue to change the Clustering or not 
            Optimised = False 
            Change_percent = 0 
            for i in Centroid_wise_data:
                Change_percent += np.sum((Centroid[i] - New_Centroid[i])*100/Centroid[i])

            if abs(Change_percent)<self.tol:
                Optimised = True
            
            if Optimised == True:
                break
    
    


