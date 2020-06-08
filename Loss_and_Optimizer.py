import numpy as np

class Loss:
    """ 
    Regularization term in the loss fucntion
    
    Parameters
    ------------------------------
    
    L2: bool, default = False
        L2 indicates the loss function whether has the regularization term 
   
    lamb: int, default = 1
       Coefficent of the regularization term 
    
    """
    def __init__(self, L2 = False, lamb = 1):
        self.lamb = lamb
        self.L2 = L2
        
class Mean_squared_error(Loss):
    """ 
    Transfer the object of class Mean_squared_error as the object of class Loss
    
    Parameters
    ------------------------------
    
    L2: bool, default = False
        L2 indicates the loss function whether has the regularization term 
   
    lamb: int, default = 1
       Coefficent of the regularization term 
    
    """    
    def __init__(self, L2 = False, lamb = 1):
        super(Mean_squared_error, self).__init__(L2, lamb)
        
class Quadratic_hinge(Loss):
    """ 
    Transfer the object of class Quadratic_hinge as the object of class Loss
    
    Parameters
    ------------------------------
    
    L2: bool, default = False
        L2 indicates the loss function whether has the regularization term 
   
    lamb: int, default = 1
       Coefficent of the regularization term 
    
    """ 
    def __init__(self, L2 = False, lamb = 1):
        super(Quadratic_hinge, self).__init__(L2, lamb)
        
class Hinge(Loss):   
    """ 
    Transfer the object of class Hinge as the object of class Loss
    
    Parameters
    ------------------------------
    
    L2: bool, default = False
        L2 indicates the loss function whether has the regularization term 
   
    lamb: int, default = 1
       Coefficent of the regularization term 
    
    """ 
    def __init__(self, L2 = False, lamb = 1):
        super(Hinge, self).__init__(L2, lamb)

class Smooth_hinge(Loss):  
    """ 
    Transfer the object of class Smooth_hinge as the object of class Loss
    
    Parameters
    ------------------------------
    
    L2: bool, default = False
        L2 indicates the loss function whether has the regularization term 
   
    lamb: int, default = 1
       Coefficent of the regularization term 
    
    """ 
    def __init__(self, L2 = False, lamb = 1):
        super(Smooth_hinge, self).__init__(L2, lamb)
        
        
        
class BaseOptimizer:
    """ 
    Base gradient descent optimizer
    
    Parameters
    ------------------------------
    learning_rate : float, default = 0.1
         The initial learning rate used. It controls the step-size in updating
         the weights.    
    """

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        
    def get_gradients(self, loss, X, y, params):
        """ 
        Get the gradient of loss function 

        Parameters
        ------------------------------
        loss: class
           It includes Mean_squared_error, Quadratic_hinge, Hinge, Smooth_hinge.
           
        X: array
         Design matrix
         
        y: array, len(y) = X.shape[0]
         Value of responsible variable
         
        params: array, len(params) = X.shape[1]
        The current parameter for supervisal learning, and it is used for updating params.
        
        Returns 
        ------------------------------
        The gradient of current loss function given loss, X, y and params.
        
        """
        
        
        import numpy as np
        
        n = len(y)
        gradient = np.zeros_like(X[0])
        
        if isinstance(loss, Quadratic_hinge):  
            if not loss.L2:
                gradient = sum(y[i]*X[i]*(y[i]*X[i].T@params - 1) for i in range(n))
                return 2*gradient/n
            else: return 2*gradient/n + [2*loss.lamb*param/n for param in params]
        
        if isinstance(loss, Mean_squared_error):  # linear
            if not loss.L2:
                gradient = (X.transpose()@(X@params - y))/X.shape[0]
                return gradient
            else: return 2*gradient/n + [2*loss.lamb*param/n for param in params]
            
        elif isinstance(loss, Hinge):
            if not loss.L2:
                gradient = sum(-y[i]*X[i] for i in range(n) if y[i]*X[i].T@params < 1)
                return gradient/n
            else: return gradient/n + [2*loss.lamb*param/n for param in params]
            
        elif isinstance(loss, Smooth_hinge):
            if not loss.L2:
                for i in range(n):
                    if y[i]*X[i].T@params <= 0:
                        gradient += -y[i]*X[i]
                    elif y[i]*X[i].T@params < 1:
                        gradient += 2*y[i]*X[i]*(y[i]*X[i].T@params - 1)
                return gradient/n
            else: return gradient/n + [2*loss.lamb*param/n for param in params]
    
    def get_loss_value(self, loss, X, y, params):
        """ 
        Get the empirical risk of the current parameter.

        Parameters
        ------------------------------
        loss: class
           It includes Mean_squared_error, Quadratic_hinge, Hinge, Smooth_hinge.
           
        X: array
         Design matrix
         
        y: array, len(y) = X.shape[0]
         Value of responsible variable
         
        params: array, len(params) = X.shape[1]
        The current parameter for supervisal learning, and it is used for updating params. 
        
        Returns 
        ------------------------------
        The empirical rish of the current parameter.
        
        """        
        
        import numpy as np
        
        n = len(y)
            
        if isinstance(loss, Mean_squared_error):
            if not loss.L2:
                return sum((X@params - y)**2)/n
        
        if isinstance(loss, Quadratic_hinge):
            value = 0
            if not loss.L2:
                for i in range(n):
                    if 1 - y[i]*X[i].T@params > 0:
                        value += (1 - y[i]*X[i].T@params)**2
                return value/n
            else:
                for i in range(n):
                    if 1 - y[i]*X[i].T@params > 0:
                        value += (1 - y[i]*X[i].T@params)**2 + loss.lamb * X[i].T @ X[i]
                    else: value += loss.lamb/n * X[i].T @ X[i]
                return value/n
                
        if isinstance(loss, Hinge):
            value = 0
            if not loss.L2:
                for i in range(n):
                    if 1 - y[i]*X[i].T@params > 0:
                        value += 1 - y[i]*X[i].T@params
                return value/n
            else:
                for i in range(n):
                    if 1 - y[i]*X[i].T@params > 0:
                        value += 1 - y[i]*X[i].T@params + loss.lamb * X[i].T @ X[i]
                    else: value += loss.lamb/n * X[i].T @ X[i]
                return value/n
        
        if isinstance(loss, Smooth_hinge):
            value = 0
            if not loss.L2:
                for i in range(n):
                    if y[i]*X[i].T@params >= 1:
                        value = value
                    elif y[i]*X[i].T@params <= 0:
                        value += 1/2 - y[i]*X[i].T@params
                    else: value += 1/2*(y[i]*X[i].T@params)**2
                return value/n
            else:
                for i in range(n):
                    if y[i]*X[i].T@params >= 1:
                        value = loss.lamb * X[i].T @ X[i]
                    elif y[i]*X[i].T@params <= 0:
                        value += 1/2 - y[i]*X[i].T@params + loss.lamb * X[i].T @ X[i]
                    else: value += 1/2*(y[i]*X[i].T@params)**2 + loss.lamb * X[i].T @ X[i] 
                return value/n
    
    def get_updates(self, loss, X, y, params):
        """
         Exception handling
        """
        raise NotImplementedError


class SGD(BaseOptimizer):
    """ 
    Stochastic gradient descent optimizer with momentum and nesterov
    
    Parameters
    ------------------------------
    learning_rate: float, default = 0.01
      The initial learning rate used. It controls the step-size in updating the weights.
      
    momentum: float, default = 0.0
      Value of momentum used, must be larger than or equal to 0
      
    nesterov: bool, default = True
      Whether to use nesterov's momentum or not. Use nesterov's if True
      
    Attributes
    ------------------------------
    velocities: list
      velocities that are used to update params
    """

    def __init__(self, learning_rate=0.01, momentum=0., nesterov=False):
        super(SGD, self).__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = []

    def get_updates(self, loss, X, y, params):
        """ 
        Get the values used to update params with given gradients

        Parameters
        ------------------------------
        loss: class
           It includes Mean_squared_error, Quadratic_hinge, Hinge, Smooth_hinge.
           
        X: array
         Design matrix
         
        y: array, len(y) = X.shape[0]
         Value of responsible variable
         
        params: array, len(params) = X.shape[1]
        Used for initializing velocities. 

        Returns
        ------------------------------
        updates: array, len(updates) = len(params)
          The values to add to params
        """
        if not self.velocities:
            self.velocities = [np.zeros_like(p) for p in params]
        if self.nesterov:
            params_for_grad = [v + p for v, p in zip(self.velocities, params)]
        else:
            params_for_grad = params
            
        grads = self.get_gradients(loss, X, y, params_for_grad)

        updates = [self.momentum * v - self.learning_rate * g
                   for v, g in zip(self.velocities, grads)]

        self.velocities = updates

        return updates

    
class Adam(BaseOptimizer):
    """
    Stochastic gradient descent optimizer with Adam
    
    Parameters
    ------------------------------
    learning_rate_init : float, default=0.001
        The initial learning rate used. It controls the step-size in updating the weights.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector, should be in [0, 1).
    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector, should be in [0, 1).
    epsilon : float, default=1e-8
        Value for numerical stability
        
    Attributes
    ------------------------------
    t : int
      Timestep  
    """
    
    def __init__(self, learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(learning_rate_init)
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        

    def get_updates(self, loss, X, y, params):
        
        """ 
        Get the values used to update params with given gradients

        Parameters
        ------------------------------
        loss: class
           It includes Mean_squared_error, Quadratic_hinge, Hinge, Smooth_hinge.
           
        X: array
         Design matrix
         
        y: array, len(y) = X.shape[0]
         Value of responsible variable
         
        params: array, len(params) = X.shape[1]
        Used for calculating gradient. 

        Returns
        ------------------------------
        updates: array, len(updates) = len(params)
          The values to add to params
        """
        
        grads = self.get_gradients(loss, X, y, params)
        
        if not hasattr(self, 'ms'):
            self.ms = [np.zeros_like(param) for param in params]
        if not hasattr(self, 'vs'):
            self.vs = [np.zeros_like(param) for param in params]
        
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        
        return updates

    
    
class Geo(BaseOptimizer):
    """
    Geometric descent optimizer
    
    Parameters
    ------------------------------
    alpha: int, default = 10
      strong convexity parameter
    
    line_search_method: {"full", "quarter", "one-third full", "one-third quarter"}, default = "full"
      It represent the way of line search, where line search is to find the minimizer of a function.
      
      -"full": for point x0 and x1, find the minimizer of a loss function in the set of hundreds of points between 2x0-x1 and 2x1-x0.
      
      -"quarter": for point x0 and x1, find the minimizer of a loss function in the set of quartiles between 2x0-x1 and 2x1-x0.
      
      -"one-third full": for point x0 and x1, find the minimizer of a loss function in the set of hundreds of points between x0 and x1
      
      -"one-third quarter": for point x0 and x1, find the minimizer of a loss function in the set of quartiles between x0 and x1.
    
    """
    def __init__(self, alpha = 10, line_search_method = 'full'):
        self.alpha = alpha
        self.line_search_method = line_search_method
    
    @staticmethod
    def _Euclidean(x_A, x_B):
        """
        Get the Euclidean distance between two points.

        Parameters
        ------------------------------
        x_A, x_B: array, x_A.shape = x_B.shape

        Returns 
        ------------------------------
        The Euclidean distance between x_A and x_B.

        """
            
        import numpy as np
        return np.linalg.norm(np.array(x_A) - np.array(x_B))
    
    def Minimum_enclosing(self, x_A, x_B, R_A, R_B):
        """
        Get a new ball enclosing the intersection of ball A and ball B.

        Parameters
        ------------------------------
        x_A, x_B: array, len(x_A) = len(x_B)
        The centers of ball A and ball B respectively. 

        R_A, R_B: float
        The radiuses of ball A and ball B respectively. 

        Returns 
        ------------------------------
        c: array
        The center of the new ball enclosing the intersection of ball A and ball B.

        R: float
        The radius of the new ball enclosing the intersection of ball A and ball B.

        """
            
        import numpy as np
        x_A = np.array(x_A)
        x_B = np.array(x_B)
        
        if Geo._Euclidean(x_A, x_B) ** 2 >= abs(R_A - R_B):
            self.c = (x_A + x_B) / 2 - (x_A - x_B) * (R_A - R_B) / (2 * (Geo._Euclidean(x_A, x_B) ** 2))
            
            R2 = R_B - (Geo._Euclidean(x_A, x_B) ** 2 + R_B - R_A) ** 2 / (4 * Geo._Euclidean(x_A, x_B) ** 2)
            self.R = R2
            
        elif Geo._Euclidean(x_A, x_B) ** 2 < (R_A - R_B):
            self.c, self.R = x_B, R_B
            
        else: self.c, self.R = x_A, R_A
        return self.c, self.R
    
    def line_search(self, x0, x1, X, y, loss, method):
        """
        Find the parameter which has the smallest loss under a specific domain related with x0 and x1 two parameters.

        Parameters
        ------------------------------
        x0, x1: array, len(x0) = len(x1) 
        The initial parameters for line search method

        X: array, len(x0) = X.shape[1]
        Design matrix

        y: array, len(y) = X.shape[0]
        Value of responsible variable

        loss: class
        It includes Mean_squared_error, Quadratic_hinge, Hinge, Smooth_hinge

        method: {"full", "quarter", "one-third full", "one-third quarter"}
        The way of line search.

        Returns 
        ------------------------------
        The parameter which has the smallest loss under a specific domain related with x0 and x1 two parameters.

        """        
        
        import numpy as np

        if method == 'full':
            searchpoll = np.linspace(2 * np.array(x0) - np.array(x1), 2 * np.array(x1) - np.array(x0), 300)
        
        elif method == 'quarter':
            searchpoll = np.linspace(2 * np.array(x0) - np.array(x1), 2 * np.array(x1) - np.array(x0), 4)
        
        elif method == 'one-third full':
            searchpoll = np.linspace(np.array(x0), np.array(x1), 100)
        
        elif method == 'one-third quarter':
            searchpoll = np.linspace(np.array(x0), np.array(x1), 4)
        
        return min(searchpoll, key = lambda s: self.get_loss_value(loss, X, y, s))
    
    

    def get_updates(self, loss, X, y, params): 
        """
        Get the values used to update params with given gradients.

        Parameters
        ------------------------------
        loss: class
        It includes Mean_squared_error, Quadratic_hinge, Hinge, Smooth_hinge

        X: array
        Design matrix

        y: array, len(y) = X.shape[0]
        Value of responsible variable

        params: array, len(params) = X.shape[1]
        Used for calculating gradient. 


        Returns 
        ------------------------------
        updates : array, len(updates) = len(params)
        The values to add to params

        """           
        
        import numpy as np
        
        method = self.line_search_method
        
        if not hasattr(self, 'beta'):
            self.beta = np.zeros_like(params)
        
        if not hasattr(self, 'c'):
            self.c = self.beta - 1 / self.alpha * self.get_gradients(loss, X, y, self.beta)
            
        if not hasattr(self, 'bata_plus'):
            self.beta_plus = self.line_search(self.beta, self.beta - self.get_gradients(loss, X, y, self.beta), X, y, loss, method)
          
        if not hasattr(self, 'R'):
            self.R = (np.linalg.norm(self.get_gradients(loss, X, y, self.beta))/self.alpha) ** 2 - \
            2/self.alpha*(self.get_loss_value(loss, X, y, self.beta) - self.get_loss_value(loss, X, y, self.beta_plus))
              
        self.beta = self.line_search(self.beta_plus, self.c, X, y, loss, method)
        self.beta_plus = self.line_search(self.beta, self.beta - self.get_gradients(loss, X, y, self.beta), X, y, loss, method)
        
        x_A = self.beta - 1 / self.alpha * self.get_gradients(loss, X, y, self.beta)
        x_B = self.c
        R_A = (np.linalg.norm(self.get_gradients(loss, X, y, self.beta))/(self.alpha)) ** 2 - \
        2 / self.alpha * (self.get_loss_value(loss, X, y, self.beta)) - self.get_loss_value(loss, X, y, self.beta_plus)            
        R_B = self.R - 2 / self.alpha * (self.get_loss_value(loss, X, y, self.beta) - self.get_loss_value(loss, X, y, self.beta_plus))  
        
        self.c, self.R = self.Minimum_enclosing(x_A, x_B, R_A, R_B)
        
        updates = self.beta - params
        
        return updates
    