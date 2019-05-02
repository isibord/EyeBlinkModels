import numpy as np
import math
import random

class Centeroid(object):
    def __init__(self, node, children):
        self.node = node
        self.children = children
        self.childrenIdx = []
        self.transitions = []
        self.transitions.append(node)
        self.closestChildIdx = 0

class KMeansModel(object):
    """A K Means Clustering Model"""
    
    def __init__(self):
        self.centeroids = []
        
    def fit(self, x, y, xRaw, kval=4, numIter=10):
        #Initialize centeroids
        self.centeroids = []
        for i in range(kval):
            randidx = random.randint(0, len(x))
            self.centeroids.append(Centeroid(x[randidx], []))
        #now iterate and update
        for i in range(numIter):
            #assign data to centeroid
            #FIRST CLEAR OUT ALL CHILDREN
            for k in range(kval):
                self.centeroids[k].children = []
                self.centeroids[k].childrenIdx = []

            for j in range(len(x)):
                # find closest centeroid
                closestcenteroid = 0
                minnorm = math.inf
                eachx = x[j]
                for k in range(kval):
                    centeroid = self.centeroids[k].node
                    centeroid = np.asarray(centeroid)
                    normval = np.linalg.norm(centeroid-eachx)
                    if normval < minnorm:
                        closestcenteroid = k
                        minnorm = normval
                self.centeroids[closestcenteroid].children.append(x[j])
                self.centeroids[closestcenteroid].childrenIdx.append(j)

            #update location of centeroid
            for k in range(kval):
                currcenteroid = self.centeroids[k]
                children = np.asarray(currcenteroid.children)
                mean = np.mean(children, axis=0)
                currcenteroid.node = mean
                currcenteroid.transitions.append(mean)


                #find closest child
                centeroidnode = np.asarray(currcenteroid.node)
                closestchild = 0
                minnorm = math.inf
                for i in range(len(currcenteroid.children)):
                    eachx = np.asarray(currcenteroid.children[i])
                    normval = np.linalg.norm(centeroidnode-eachx)
                    if normval < minnorm:
                        minnorm = normval
                        closestchild = currcenteroid.childrenIdx[i]
                currcenteroid.closestChildIdx = closestchild


        
        #print everything needed for plot
        for k in range(kval):
            #everything for each centeroids
            currcenteroid = self.centeroids[k]
            print(k, " - Transitions for Centeroid")
            for trans in currcenteroid.transitions:
                print(trans[0], trans[1])

            print(k, " - Closest Child ", xRaw[currcenteroid.closestChildIdx])

            print("")
            print(k, " - Children for Centeroid")
            for child in currcenteroid.children:
                print(child[0], child[1])
            print("")

            


    def predict(self, x):
        # for every sample in training set
        # compute the distance to xText (using l2 norm)
        pass