# STA208-Project
Our project theme is the implementation of "The geometric alternative to Nestorov's accelerated algorithm"(GeoD).  
The main components in our project can be divided into the following parts:  
1. Implementation of GeoD   
2. Build class for loss functions which contains 4 different kinds which vary as respect to convexness.
Specificall, we select Quadratic, Exponential, Hinge and Smooth_hinge loss types.  
3. We compare the iteration performance of GeoD visually with other gradient descent algorithms of Classical Momentum Gradient Descent,
Nesterov's Gradient Descent, Stochaostic Gradient Descent and Adam Gradient Descent. All of these gradient algorithms are built under the same framework but not GeoD considering its unique interation structure.

Progress most recently:  
We have finished the implementation of GeoD and all the other algorithms. Also, the loss funtion class is finished as well.  
Next step is to try to merge GeoD under the same framework and make the comparisons visually.
