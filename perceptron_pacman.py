# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.


        # ({'West': {'foodCount': 95}, 'East': {'foodCount': 96}, 'Stop': {'foodCount': 96}}, ['West', 'Stop', 'East'])

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)): #data = states 
                "*** YOUR CODE HERE ***"
                update = util.Counter()
                actions = trainingData[i][1]
                features = trainingData[i][0]
                for action in actions:
                    feature = features[action]
                    update[action] = self.weights*feature
                maxaction = update.argMax()
                if (maxaction == trainingLabels[i]):
                    continue
                
                self.weights += features[trainingLabels[i]]
                self.weights -= features[maxaction]
                #print(str(key) + " prev weight: " + str(self.weights[key]))
                #print("added: " + str(trainingLabels[i]) +" " + str(features[trainingLabels[i]][key])) 
                #print("sub: " + str(maxaction) + " " + str(maxfeat[key]))  
            
                #print(str(key) + ": " + str(self.weights[key]))


 
