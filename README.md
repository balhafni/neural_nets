# Simple Neural Networks Examples:

This repo contains just three simple, yet very interesting examples of neural nets.

### Setup:

To run the examples in this repo, you would need to install conda.

After that just follow these simple steps:

Clone the repo to your local machine
```
git clone https://github.com/balhafni/neural_nets.git
```
Install all the dependencies you need by using the envrionment.yaml file
```
conda env create -f environment.yml
```

## 1) Logistic Regression with a Neural Net mindset: 
A simple logistic regression classifier to recognize images of cats with up to 70% accuracy.

Detecting a cat:
![cat](https://github.com/balhafni/neural_nets/blob/master/log_reg_mindset/results/cat.png)

Detecting a non-cat:
![non-cat](https://github.com/balhafni/neural_nets/blob/master/log_reg_mindset/results/non-cat.png)

## 2) Planar Classifier:
A simple classifier that uses gradient descent and only a single hidden layer. 

Random blue and red points on a plane:
![random points](https://github.com/balhafni/neural_nets/blob/master/planar_classifier/results/planar_rand_points.png)

Classified points:
![classified points](https://github.com/balhafni/neural_nets/blob/master/planar_classifier/results/planar_after.png)

## 3) Sign Digits Recognizer:
An optimized deep neural net that uses mini-batch gradient descent to recognize sign digits from 0 to 5 (using Tensorflow YAY!)
