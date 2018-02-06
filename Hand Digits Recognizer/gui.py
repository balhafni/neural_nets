import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import h5py
import scipy
from scipy import ndimage
from PIL import Image
import tkinter
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import random
import signs_recognizer as recognizer
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

class gui:
    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("Signs Dataset Neural Net")
        self.window.geometry("300x50")
        self.frame = None
        self.msg = None
        self.canvas = None
        self.figure = None
        self.a = None  
        self.parameters = None
        self.button_train = tkinter.Button(self.window, text = 'train model', command = self.train)
        self.button_train.pack()
      
        self.window.mainloop()
    
    def train(self):
        recognizer.load_data()
        print('Training....')
        X_train, Y_train, X_test, Y_test = recognizer.load_data()
        self.parameters = recognizer.model(X_train, Y_train, X_test, Y_test)
        print('Training is DONE!')
        self.load_image_frame()

    def load_image_frame(self):
        #creating a new frame for the image
        self.frame = tkinter.Tk()
        self.frame.title("Resutls")
        self.msg =  tkinter.Label(self.frame)
        self.msg.pack()
        self.figure = Figure(figsize = (5,5), dpi = 100)
        self.a = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure,self.frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack()
        button_choose = tkinter.Button(self.frame, text = 'Pick your own image', command = self.get_chosen_image)
        button_choose.pack()
        self.frame.mainloop()

    def get_chosen_image(self):
        fileName = askopenfilename()
        try:
            image = np.array(ndimage.imread(fileName, flatten=False))
            my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
            my_image_prediction = predict(my_image, self.parameters)
            self.a.imshow(image)
            self.canvas.draw()
            self.msg["text"] = "Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction))
        except Exception as e:
            tkinter.messagebox.showerror("Error",e)

if __name__ =='__main__':
    g = gui()