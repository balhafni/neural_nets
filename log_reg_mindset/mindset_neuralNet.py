import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import h5py
import scipy
from scipy import ndimage

import tkinter
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import random

class Mindset_NeuralNet:
    def __init__(self):
        self.train_set_x_orig = None
        self.train_set_y_orig = None
        self.test_set_x_orig = None
        self.test_set_y_orig = None
        self.classes = None
        self.train_set_x = None
        self.test_set_x = None
        self.train_set_y = None
        self.test_set_y = None
        self.m_train = 0
        self.m_test = 0
        self.num_px = 0

    def load_dataset(self):
        train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
        self.train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #  train set features
        self.train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #  train set labels

        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        self.test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
        self.test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set label
        self.classes = np.array(test_dataset["list_classes"][:]) # the list of classes
   
        self.train_set_y = self.train_set_y_orig.reshape((1, self.train_set_y_orig.shape[0]))
        self.test_set_y = self.test_set_y_orig.reshape((1, self.test_set_y_orig.shape[0]))
    
      

    def preprocess_data(self):
        self.m_train = self.train_set_x_orig.shape[0] # Number of training examples
        self.m_test = self.test_set_x_orig.shape[0] # Number of testing examples:
        self.num_px = self.train_set_x_orig[0].shape[0] # Height/Width of each image

        ''' Reshaping the training and test data sets so that images of size (num_px, num_px, 3) 
        are flattened into single vectors of shape (num_px ∗ num_px ∗ 3, 1)'''
        train_set_x_flatten = self.train_set_x_orig.reshape(self.m_train,-1).T 
        test_set_x_flatten = self.test_set_x_orig.reshape(self.m_test,-1).T

        '''stadarized datasets '''
        self.train_set_x = train_set_x_flatten/255.
        self.test_set_x = test_set_x_flatten/255.


    def sigmoid(self,z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """
        s = 1 / (1 + np.exp(-z))
    
        return s

    def initialize_with_zeros(self,dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
    
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros((dim  , 1))
        b = 0

        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
    
        return w, b

    def propagate(self,w, b, X, Y):
        """
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
    
        m = X.shape[1]
    
        # FORWARD PROPAGATION (FROM X TO COST)
        z = np.dot(w.T,X) + b
        A = self.sigmoid(z)                              # compute activation
        cost =  (-1/m) * np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)))       # compute cost
    
        # BACKWARD PROPAGATION (TO FIND GRAD)
   
        dz = A - Y
        dw = (1/m) * np.dot(X,dz.T)
        db = (1/m) * np.sum(dz)
  

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
    
        grads = {"dw": dw,
                 "db": db}
    
        return grads, cost


    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
    
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
    
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
    
        costs = []
    
        for i in range(num_iterations):
        
        
            # Cost and gradient calculation
     
            grads, cost = self.propagate(w,b,X,Y)
        
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
        
            # update rule
            w = w - learning_rate * dw
            b = b - learning_rate * db
  
        
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
        
            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
    
        params = {"w": w,
                  "b": b}
    
        grads = {"dw": dw,
                 "db": db}
    
        return params, grads, costs


    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
    
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
    
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
    
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = self.sigmoid(np.dot(w.T,X) + b)
    
        for i in range(A.shape[1]):
        
            # Convert probabilities A[0,i] to actual predictions p[0,i]
  
            if A[0,i] <= 0.5:
                Y_prediction[0,i] = 0
            else:
                Y_prediction[0,i] = 1
     
    
        assert(Y_prediction.shape == (1, m))
    
        return Y_prediction



def model(m, X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    
    # initialize parameters with zeros    
    w, b = m.initialize_with_zeros(X_train.shape[0])
    
    # Gradient descent
    parameters, grads, costs = m.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = m.predict(w, b, X_test)
    Y_prediction_train = m.predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d




class GUI:
    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("Mindset Neural Net")
        self.window.geometry("300x50")
        self.frame = None
        self.msg = None
        self.canvas = None
        self.figure = None
        self.a = None
        self.mindset = Mindset_NeuralNet()
        self.d = {}
       
        self.button_train = tkinter.Button(self.window, text = 'train model', command = self.train)
        self.button_train.pack()
      
        self.window.mainloop()

    def train(self):
        self.mindset.load_dataset()
        self.mindset.preprocess_data()
        print('Training....')
        self.d = model(self.mindset, self.mindset.train_set_x, self.mindset.train_set_y, self.mindset.test_set_x, self.mindset.test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
        #creating a new frame for the image
        self.frame = tkinter.Tk()
        self.frame.title("Resutls")
        self.msg =  tkinter.Label(self.frame)
        self.msg.pack()
        self.figure = Figure(figsize = (5,5), dpi = 100)
        self.a = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure,self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        button_test = tkinter.Button(self.frame, text = 'Pick a random image from the dataset', command = self.get_random_image)
        button_test.pack()
        # button_choose = tkinter.Button(self.frame, text = 'Pick your own image', command = self.get_chosen_image)
        # button_choose.pack()
        self.frame.mainloop()

    def get_chosen_image(self):
        fileName = askopenfilename()
        try:
            image = np.array(ndimage.imread(fileName, flatten=False))
            my_image = scipy.misc.imresize(image, size=(self.mindset.num_px,self.mindset.num_px)).reshape((1, self.mindset.num_px*self.mindset.num_px*3)).T
            my_predicted_image = self.mindset.predict(self.d["w"], self.d["b"], my_image)
            self.a.imshow(image)
            self.canvas.draw()
            self.msg["text"] = "y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + self.mindset.classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture."
        except Exception as e:
            tkinter.messagebox.showerror("Error",e)



    def get_random_image(self):
        '''
        Getting a random image 
        '''
        index = random.randint(0,49) 
        #Plotting the image
      
        #self.msg["text"] = self.mindset.classes[self.d["Y_prediction_test"][0,index]].decode("utf-8")
        
        self.a.imshow(self.mindset.test_set_x[:,index].reshape((self.mindset.num_px, self.mindset.num_px, 3)))
        self.canvas.draw()

        self.msg["text"] = "y = " + str(self.mindset.test_set_y[0,index]) + ", you predicted that it is a \"" + self.mindset.classes[int(self.d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.";
       
        #plt.show()
        #print ("y = " + str(self.mindset.test_set_y[0,index]) + ", you predicted that it is a \"" + self.mindset.classes[self.d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
       



if __name__ == '__main__':
   frame = GUI()
   # m = Mindset_NeuralNet()
   # m.load_dataset()
   # m.preprocess_data()
   # d = model(m, m.train_set_x, m.train_set_y, m.test_set_x, m.test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
   # index = 10
   ## print(str(int(d["Y_prediction_test"][0,index])))
   # plt.imshow(m.test_set_x[:,index].reshape((m.num_px, m.num_px, 3)))
   # print ("y = " + str(m.test_set_y[0,index]) + ", you predicted that it is a \"" + m.classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")