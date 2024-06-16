# Coding the Linear Rerression Model 

import numpy as np

# Define th Inputs

print("This python file contains algorithem for Lasso regression")

class Lasso_Regression:
    def __init__(self,lr,maxiters,lasso_pen):
        """Initialize the Lasso Regression Model

        Args:
            lr (_type_): learninig Rate
            maxiters (_type_): Max Count of iterations
            lasso_pen (_type_): L1 Reglularization Penalty
        """
        print("Initailizing the Lasso Regression Class")
        
        self.learning_rate = lr
        self.maxiters = maxiters
        self.lasso_pen = lasso_pen

    
    def Lasso_Regression_CF(self,Xtrain,Ytrain,weights,bias,n_datapoints,lasso_pen):  
        
        """Computes the Weights and Bias Gradients for the Lasso REgression Cost functions 

        Args:
            Xtrain (_type_): Training Data Features
            Ytrain (_type_): Traininig Data Target
            weights (_type_): Regression weights
            bias (_type_): Regression Bias
            n_datapoints (_type_): No of datapoints
            lasso_pen (_type_): L1 Regulaization Penalty

        Returns:
            _type_: Gradients of Lasso regression cost functions
        """    
        
        ypred = self.predict(self.Xtrain)
        
        # Estimating the Gradients for Weights and Bias 
        # Weights
        dw  = np.zeros(self.n_features) 
        
        for i in range(self.n_features):
                        
            if weights[i]>0:
                dw[i] = (2/n_datapoints)*(np.dot(Xtrain[:,i].T,(ypred - Ytrain) - self.lasso_pen))
            else:
                dw[i] = (2/n_datapoints)*(np.dot(Xtrain[:,i].T,(ypred - Ytrain) + self.lasso_pen))
        
        # Bias
        db = (2/n_datapoints)*np.sum(ypred - Ytrain)
        
        return dw,db
    
    def fit(self,X,Y):
        
        self.Xtrain  = X.to_numpy()
        self.Ytrain = Y.to_numpy()
        
        self.n_datapoints, self.n_features = self.Xtrain.shape
        
        #Initailizing Weights and biases 
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        
        # Gradient Descent Algorithem
        # Run the Iteration to calcualte and optimixe the Weights nand bias based on CF values
        for i in range(self.maxiters):
            
            dw,db = self.Lasso_Regression_CF(self.Xtrain, self.Ytrain,self.weights,self.bias,self.n_datapoints,self.lasso_pen)
            
            print("Udpating the Weights and bias now")
            #Updating the Weights
            self.weights -= self.learning_rate*dw         
            #Updating the new bias
            self.bias -= self.learning_rate*db
            
            print(f"weights {self.weights} and bias {self.bias}")
            
        print(f"The Function is fit with the weights {self.weights} and bias {self.bias} with iterations of {self.maxiters}")   
        
    def predict(self,X):
        
        Ypred = np.dot(X,self.weights) + self.bias
        
        return Ypred
