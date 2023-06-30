# GridCellPredNet

This repository accompanies the paper "Ring Attractors as the Basis for a Biomimetic Navigation System" Thomas C. Knowles, Anna Summerton, James G.H. Whiting and Martin J. Pearson.

--- link pending ---

## The model in action:

![alt text](https://github.com/TomKnowles1994/GridCellPredNet/blob/main/corrected_vs_uncorrected_short.gif "Animated Ring Attractor Visualisation")

This animation illustrates the model produced. It consists of three spiking Ring Attractors that together model a robot's planar translation. This is supported by sensory data from the robot, synthesised into multisensory `experiences' by a Predictive Coding Network (PCN). The uncorrected model (green error line) can track the trajectory, but is subject to drift. The PCN-corrected model can use sensory data to compensate for drift, by recalling these prior experiences and their locations. Each Cartesian coordinate maps to a given ring state, and vice-versa, providing targets on the rings for corrective input.
