import tkinter
import planar_classifier as classifier
import planar_utils
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset, plot_decision_boundary

class gui:
    def __init__(self):
        self.X = None
        self.Y = None
        self.window = tkinter.Tk()
        self.window.title("Planar Data Classifier")
        self.window.geometry("300x100")
        self.loadBtn = tkinter.Button(self.window, text = 'Load Planar Model',command = self.load_data)
        self.loadBtn.pack()

        self.window.mainloop()

    def load_data(self):
        plt.close()
        self.X,self.Y = load_planar_dataset()
        self.trainBtn = tkinter.Button(self.window, text = 'Train Model',command = self.train_data)
        self.trainBtn.pack()
        plt.scatter( self.X[0, :], self.X[1, :], c=self.Y.ravel(), s=40, cmap=plt.cm.Spectral)
        plt.show()

    def train_data(self):
        parameters = classifier.nn_model(self.X, self.Y, n_h = 4, num_iterations = 10000, print_cost=True)
        plot_decision_boundary(lambda x: classifier.predict(parameters, x.T), self.X, self.Y.ravel())
        self.trainBtn.pack_forget()
        plt.title("Decision Boundary for hidden layer size " + str(4))
        plt.show()


if __name__ == '__main__':
    frame = gui()