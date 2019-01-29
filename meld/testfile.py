# quick demo of meldconvex.meld()
# dependencies: pygsp, numpy, matplotlib, graphtools

import pygsp
import numpy as np
import sys
import meldconvex
import matplotlib.pyplot as plt
#edit this line to your graphtools path (or import it in a cleaner way.)
sys.path.append('/storage/External/experiments/graphtools')
import graphtools


fig1, ax = plt.subplots(2,2)
g = pygsp.graphs.Ring() #dummy graph to show
signal = (g.coords>0).astype(int) #slice by values to create a 'timepoint'
g.plot_signal(signal[:,0], ax = ax[0,0])
g.plot_signal(signal[:,1], ax = ax[0,1])
output = meldconvex.meld(signal,10,g) 
# output will be the output of numpy lsq, with residuals etc.   take the first slice.
meldoutput = output[0]
g.plot_signal(meldoutput[:,0], ax = ax[1,0])
g.plot_signal(meldoutput[:,1], ax = ax[1,1])
plt.savefig('ringdemo.png', format = 'png')
# let's do this with an instance of a graphtools graph object.

inputs = np.random.normal(size=[100,2]) # you can substitute your own data here.
fig2, ax = plt.subplots(2,2)

g = graphtools.Graph('knn', inputs) 
g.set_coordinates(inputs)
signal = (g.coords>0).astype(int) #slice by values to create a 2D 'timepoint' vector
g.plot_signal(signal[:,0], ax = ax[0,0])
g.plot_signal(signal[:,1], ax = ax[0,1])
output = meldconvex.meld(signal,10,g) 
# output will be the output of numpy lsq, with residuals etc.   take the first slice.
meldoutput = output[0]
g.plot_signal(meldoutput[:,0], ax = ax[1,0])
g.plot_signal(meldoutput[:,1], ax = ax[1,1])
plt.savefig('gaussiandemo.png', format = 'png')