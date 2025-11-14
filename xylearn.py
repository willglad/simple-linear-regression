"""
-----------------------------------------------------------------------------------------------

LINEAR REGRESSION PROGRAM - FOR X Y DATASETS ONLY

-----------------------------------------------------------------------------------------------

"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Model:

    def __init__(self, b0, w):

        #initialise bias and weight parameters, and loss list
        self.params = { "b0" : b0,
                "w": w}
        self.loss = []

    def getParams(self):
        #returns parameters
        return self.params
    
    def unnormaliseParams(self, x_mean, x_std, y_mean, y_std):
        # Store original normalized parameters
        w_norm = self.params["w"]
        b0_norm = self.params["b0"]

        # Unnormalize weight and bias
        w_orig = w_norm * (y_std / x_std)
        b0_orig = y_mean + y_std * b0_norm - w_orig * x_mean

        self.params["w"] = w_orig
        self.params["b0"] = b0_orig
        
    def calculateMSE(self, data):

        x = np.array(data["x"]) #separate x and y variables
        y = np.array(data["y"])

        #gets the fitted y values
        y_fit = x * self.params["w"] + self.params["b0"] 

        #calcs difference between y values and the fitted y values   
        diff = y - y_fit

        #get the squared error
        SE = np.sum(diff ** 2)

        #return Mean Squared Error
        return SE / len(x)
    
    def calcLossDerivs(self, data):

        #separate x and y data into numpy arrays
        x = np.array(data["x"])
        y = np.array(data["y"])
        y_fit = x * self.params["w"] + self.params["b0"] # calc y fit values

        b0 = self.params["b0"]
        w = self.params["w"]

        db0 = -2 * np.mean(y - (w * x + b0))    #derivative of b0
        dw  = -2 * np.mean((y - (w * x + b0)) * x)  #derviative of w

        return [db0, dw]
    
    def incParams(self, data, learn_rate):

        d = self.calcLossDerivs(data) #gets loss derivatives

     
        self.params["b0"] = float(self.params["b0"] - learn_rate * d[0]) # decrease each parameter by the respective derivative * learn rate (direction preserved in d array) 
        self.params["w"] = float(self.params["w"] - learn_rate * d[1])
        
        return
    
    def learn(self, data, learn_rate, iterations):
        #reset loss list
        self.loss = []

        for _ in range(iterations):
            self.incParams(data, learn_rate) #increment parameters for a single iteration
            self.loss.append(self.calculateMSE(data)) # add the loss value to a list for plotting purposes
        
        if self.loss[-1] > self.loss[0]:
            print("WARNING: Loss increased during training. Try altering initial parameters and learning rate.") #warn user if loss did not decrease

        return self.loss
    
    def plotGraphs(self, data, plot_title, plot_xlabel, plot_ylabel):

        x = data["x"]
        y = data["y"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5))

        #Plot data 
        
        ax1.scatter(x, y, color = "blue" , label = "Data")

        #Plot regression line
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = self.getParams()["b0"] + self.getParams()["w"] * x_vals
        ax1.plot(x_vals, y_vals, color = "red" , label = "Model")
        ax1.set_title(plot_title)
        ax1.set_xlabel(plot_xlabel)
        ax1.set_ylabel(plot_ylabel)
        ax1.legend()

        #Create loss plot
        ax2.plot(self.loss)
        ax2.set_title("Loss Plot")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Normalised Loss (MSE)")
        
        #Show both data + model, and loss plot seperately
        plt.tight_layout()
        plt.show()

def get_float(prompt, default):
    try:
        return float(input(prompt))
    except Exception:
        print(f"Invalid input. Using default: {default}")
        return default

def get_int(prompt, default):
    try:
        return int(input(prompt))
    except Exception:
        print(f"Invalid input. Using default: {default}")
        return int(default)

def train_model(plot_query, data = {"x" : [], "y" : []}):
    #get data statistics for normalisation
    x_mean = data["x"].mean()
    x_std = data["x"].std()
    y_mean = data["y"].mean()
    y_std = data["y"].std()
    print(type(x_mean))

    #normalise data
    data["x"] = (data["x"] - x_mean) / x_std
    data["y"] = (data["y"] - y_mean) / y_std

    b0 = get_float("Enter bias estimate: ", 0)
    w = get_float("Enter weight estimate: ", 0)
    learning_rate = get_float("Enter learning rate: ", 0.01)
    iterations = get_int("Enter number of iterations: ", 1000)

    model = Model(b0, w)
    model.learn(data, learning_rate, iterations)

    #unnormalise so that data makes sense
    data["x"] = (data["x"] * x_std) + x_mean
    data["y"] = (data["y"] * y_std) + y_mean

    #unnormalise model parameters
    model.unnormaliseParams(x_mean, x_std, y_mean, y_std)

    if plot_query: model.plotGraphs(data, "Title", "X", "Y")


    

