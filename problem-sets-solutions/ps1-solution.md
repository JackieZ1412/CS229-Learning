# CS229 ps1-solution

## Problem 1

### Sub Problem (a)

For the loss function $J(\theta)$，the element of the hessian matrix is：
$$
H_{ij} = \frac{\part^{2}J(\theta)}{\part\theta_i\part\theta_j}
$$
So for the specific loss function $J(\theta)$
$$
J(\theta) = -\frac{1}{m}\sum_{i = 1}^{m}y^{(i)}log(h_{\theta}(x^{(i)}))\, + \,  (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)}))
$$
We got:
$$
\frac{\part{J(\theta)}}{\part\theta_i} = -\frac{1}{m}\sum_{k = 1}^{m}(x_{i}^{(k)}y^{(k)} - x_{i}^{(k)}g(\theta^{T}x))
$$
So the second derivative is:
$$
\frac{\part^{2}J(\theta)}{\part{\theta_i}{\theta_j}} = \sum_{k = 1}^{m}\frac{x_{i}^{(k)}x_{j}^{(k)}}{m}g(\theta^{T}x)(1 - g(\theta^{T}x))
$$
We want to prove $z^{T}Hz \geq 0$：
$$
z^THz = \sum_{k = 1}^{m}\sum_{i = 1}^{m}\sum_{j = 1}^{m}\frac{z_{i}x_{i}^{(k)}x_{j}^{(k)}z_{j}}{m}g(\theta^{T}x)(1 - g(\theta^{T}x))
$$
Consider the sum, we got:
$$
\sum_{i = 1}^{m}\sum_{j = 1}^{m}z_{i}x_{i}x_{j}z_{j} = (xz)^{T}(xz) \geq 0
$$
the resident part $0 < g(\theta^Tx) < 1$，so we finish the proof.

### Sub Problem (b)

```python
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)
    util.plot(x_train, y_train,model.theta,'output/p01b.png')
    x_out,y_out = util.load_dataset(eval_path, add_intercept=True)
    prediction = model.predict(x_eval)
    np.savetxt(pred_path,prediction > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        self.theta = np.zeros(n)
        while True:
            old_theta = np.copy(self.theta)
            hx = 1 / (1 + np.exp(-x.dot(self.theta)))
            hessian = (x.T * hx * (1 - hx)).dot(x) / m
            gradient_J_theta = x.T.dot(hx - y) / m
            self.theta = self.theta - np.linalg.inv(hessian).dot(gradient_J_theta)
            if np.linalg.norm(self.theta - old_theta,ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / 1 + np.exp(-x.dot(self.theta))
        # *** END CODE HERE ***

```

Use Newton's method to solve the problem.

### Sub Problem (c)

First，by bayes formula：
$$
p(y = 1 | x) = \frac{p(x|y = 1)p(y = 1)}{p(x|y = 1)p(y = 1) + p(x|y = 0)p(y = 0)}
$$
Bring in the possibility of the defination，we got：
$$
p(y = 1|x) = \frac{1}{1 + \frac{1 - \phi}{\phi}exp\{\frac{1}{2}((x - \mu_0)^{T}\sum^{-1}(x - \mu_0) - (x - \mu_1)^{T}\sum^{-1}(x - \mu_1))\}}
$$
Simplify the formula：
$$
p(y = 1|x) = \frac{1}{1 + exp\{\sum^{-1}(\mu_0 - \mu_1)^Tx + \frac{1}{2}((\mu_0 + \mu_1)^T\sum^{-1}(\mu_1 - \mu_0)) - log(\frac{1 - \phi}{\phi})\}}
$$
So，define the parameter like this：
$$
\theta = (\mu_0 - \mu_1)(\sum)^{-1}
$$
**note that** $\sum^{-1}$ **is a symmetric matrix**
$$
\theta_0 = \frac{1}{2}((\mu_0 + \mu_1)^T\sum^{-1}(\mu_1 - \mu_0)) - log(\frac{1 - \phi}{\phi})
$$
So $p(y = 1|x)$ can be represented by the formula as the question displays.

### Sub Problem (d)

$$
\ell(\phi,\mu_0,\mu_1,\sum) = log\prod_{i = 1}^{m}p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\sum)\\= log\prod_{i = 1}^{m}p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\sum)p(y^{(i)};\phi) \\
= \sum_{i = 1}^{m}(log(p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\sum)) + log(p(y^{(i)};\phi)))
$$

We calculate the $\phi$ first：
$$
\frac{\part \ell}{\part \phi} = \sum_{i = 1}^{m}y^{(i)}\frac{\part log\phi}{\part\phi} + (1 - y^{(i)})\frac{\part log(1 - \phi)}{\part \phi} \\
= \sum_{i = 1}^{m}\frac{y^{(i)}}{\phi} - \frac{y - \phi}{1 - \phi}\\
 = \frac{(\sum_{i = 1}^{m}y^{(i)}) - m\phi}{\phi(1 - \phi)} = 0
$$
So make the derivative maximum，we have to let：
$$
\phi = \frac{\sum_{i = 1}^{m}I(y^{(i)} = 1)}{m}
$$
Then we calculate $\mu_0,\mu_1,\sum$：
$$
\frac{\part \ell}{\part \mu_0} = \frac{\part \ell}{\part \mu}\frac{\part \mu}{\part \mu_0}\\
= -\frac{1}{2\sqrt{2\pi}}\sum_{i = 1}^{m}\frac{\part (x^{(i)} - \mu)^T\sum^{-1}(x^{(i)} - \mu)}{\part \mu}\frac{\part \mu}{\part \mu_0}\\
= -\frac{\sum^{-1}}{\sqrt{2\pi}}\sum_{i = 1}^{m} I(y^{(i)} = 0)(x^{(i)} - \mu_0) \\ 
 = -\frac{1}{\sqrt{2\pi}}\sum_{i = 1}^{m}I(y^{(i)} = 0)(x^{(i)} - \mu_0) = 0
$$
So，we can calculate $\mu_0$：
$$
\mu_0 = \frac{\sum_{i = 1}^{m}I(y^{(i)} = 0)x^{(i)}}{\sum_{i = 1}^{m}I(y^{(i)} = 0)}
$$
For the same derivation，we know that $\mu_1$ is：
$$
\mu_1 = \frac{\sum_{i = 1}^{m}I(y^{(i)} = 1)x^{(i)}}{\sum_{i = 1}^{m}I(y^{(i)} = 1)}
$$
At last，we calculate the covariance matrix $\sum^{-1}$
$$
\frac{\part \ell}{\part \sum} = \frac{m}{2}(\sum)^{-1} + \frac{1}{2}(\sum)^{-1}(\sum_{i = 1}^{m}(x - \mu_{y^{(i)}})(x - \mu_{y^{(i)}})^T)(\sum)^{-1} = 0
$$
So the covariance matrix is：
$$
\sum = \frac{\sum_{i = 1}^{m}(x - \mu_{y^{(i)}})(x - \mu_{y^{(i)}})^T}{m}
$$
Proof finished.

### Sub Problem (e)

```python
from statistics import covariance
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)
    util.plot(x_train, y_train,model.theta,'output/p01e.png')
    x_out,y_out = util.load_dataset(eval_path, add_intercept=True)
    prediction = model.predict(x_out)
    np.savetxt(pred_path,prediction > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        phi = sum(y == 1) / m
        u0 = np.sum(x[y == 0],axis=0) / (m - sum(y == 1))
        u1 = np.sum(x[y == 1],axis=0) / sum(y == 1)
        Covariance = ((x[y == 0] - u0).T.dot(x[y == 0] - u0) + (x[y == 1] - u1).T.dot(x[y == 1] - u1))
        inverse_covariance = np.linalg.inv(Covariance)
        self.theta = np.zeros(n + 1)
        self.theta[0] = (u0 + u1).dot(inverse_covariance).dot(u0 - u1) * 0.5
        self.theta[1:] =  inverse_covariance.dot(u1 - u0)
        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE

```

Use GDA parameter analysis to write the fit function, and it is the same with the linear analysis in sub problem (b) in main.

**Remember to give theta the value by rules from the notes**

### Sub Problem (f)



### Sub Problem (g)



### Sub Problem (h)



## Problem 2

### Sub Problem (a)

$$
p(y = 1,t = 1,x) = p(y = 1|t = 1,x)p(t = 1 | x)p(x)\\
= p(t = 1|y = 1)p(y = 1|x)p(x)
$$

From the known factors：
$$
p(t = 1|y = 1) = 1\\
p(y = 1|t = 1,x) = p(y = 1|t = 1)
$$
So we can derivate：
$$
\frac{p(t = 1|x)}{p(y = 1|x)} = \frac{p(t = 1|y = 1,x)}{p(y = 1|t = 1,x)} = \frac{1}{p(y = 1 | t = 1)}
$$
From the problem we know it is a constant.

### Sub Problem (b)

From sub problem (a) we know：
$$
h(x) \approx p(y = 1|x) \approx \alpha
$$

### Sub Problem (c,d,e)

```python
import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    #######################################################################################
    # Problem (c)
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model_t = LogisticRegression()
    model_t.fit(x_train, t_train)

    util.plot(x_test, t_test, model_t.theta, 'output/p02c.png')

    t_pred_c = model_t.predict(x_test)
    np.savetxt(pred_path_c, t_pred_c > 0.5, fmt='%d')
    #######################################################################################
    # Problem (d)
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    model_y = LogisticRegression()
    model_y.fit(x_train, y_train)

    util.plot(x_test, y_test, model_y.theta, 'output/p02d.png')

    y_pred = model_y.predict(x_test)
    np.savetxt(pred_path_d, y_pred > 0.5, fmt='%d')
    #######################################################################################  
    # Problem (e)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    alpha = np.mean(model_y.predict(x_valid))

    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, 'output/p02e.png', correction)

    t_pred_e = y_pred / alpha
    np.savetxt(pred_path_e, t_pred_e > 0.5, fmt='%d')
    #######################################################################################
    # *** END CODER HERE

```



## Problem 3

### Sub Problem (a)

Poisson distribution can be written as：
$$
p(y;\lambda) = \frac{1}{y!}exp\{ylog\lambda - \lambda\}
$$
For the parameters in exponential family：
$$
b(y) = \frac{1}{y!}\\
\eta = log\lambda\\
T(y) = y \\
\alpha(\eta) = \exp\{\eta\}
$$
Proof finished.

### Sub Problem (b)

$$
h_{\theta}(x) = \lambda = exp\{\theta^Tx\}
$$

### Sub Problem (c)

$$
log\,p(y^{(i)}|x^{(i)};\theta) = log\,\frac{1}{y^{(i)}!}exp\{\theta^Tx^{(i)}y^{(i)} - exp\{\theta^Tx^{(i)}\}\} \\
$$

So we take the derivative of $\theta_j$：
$$
\frac{\part log\,p(y^{(i)}|x^{(i)};\theta)}{\part \theta_j} = x_{j}^{(i)}y^{(i)} - x_{j}^{(i)}exp\{\theta^Tx^{(i)}\}
$$
The adjustment to the $\theta_j$ is：
$$
\theta_j = \theta_j + \alpha(x_{j}^{(i)}y^{(i)} - x_{j}^{(i)}exp\{\theta^Tx^{(i)}\})
$$

### Sub Problem (d)

```python
import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = PoissonRegression(step_size=lr,eps=1e-5)
    model.fit(x_train, y_train)
    x_out,y_out = util.load_dataset(eval_path, add_intercept=True)
    prediction = model.predict(x_out)
    np.savetxt(pred_path,prediction > 0.5, fmt='%d')
    plt.figure()
    plt.plot(y_out,prediction,'bx')
    plt.xlabel('True data')
    plt.ylabel('Predict data')
    plt.savefig('p03d.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        self.theta = np.zeros(n)
        while True:
            theta = np.copy(self.theta)
            self.theta += self.step_size * (x.T.dot(y) - x.T.dot(np.exp(x.dot(self.theta))))
            if np.linalg.norm(self.theta - theta,ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***

```



## Problem 4

### Sub Problem (a)

Start with the hint：
$$
\frac{\part}{\part \eta} \int p(y;\eta)dy = \int \frac{\part}{\part \eta}p(y;\eta)dy = 0\\
\int \frac{\part}{\part \eta}p(y;\eta)dy = \int b(y)(T(y) - \frac{\part \alpha(\eta)}{\part \eta})exp\{\eta^TT(y) - \alpha(\eta)\} \\
= \int (T(y) - \frac{\part \alpha(\eta)}{\part \eta})p(y;\eta)dy\\
T(y) = y \\
\int yp(y;\eta)dy = \frac{\part \alpha(\eta)}{\part \eta}\int p(y;\eta)dy = \frac{\part \alpha(\eta)}{\part \eta}\\
E(Y|X;\eta) = E(Y;\eta) = \frac{\part \alpha(\eta)}{\part \eta}
$$
Proof finished.

### Sub Problem (b)

From the hint in sub problem (a) and the sub problem (a) itself，we can derivate：
$$
\frac{\part}{\part \eta}\int yp(y;\eta)dy = \frac{\part^2}{\part \eta^2}\alpha(\eta) \\
\frac{\part}{\part \eta}\int yp(y;\eta)dy = \int \frac{\part}{\part \eta}(yp(y;\eta))dy \\
 = \int y(y\, - \,\frac{\part \alpha(\eta)}{\part \eta})p(y;\eta)dy \\
 = E(Y^2;\eta) - (E(Y;\eta))^2 \\
 = \frac{\part^2}{\part \eta^2}\alpha(\eta)
$$
So the variance is：
$$
Var(Y|X;\eta) = Var(Y;\eta) = E(Y^2;\eta) - (E(Y;\eta))^2 
 = \frac{\part^2}{\part \eta^2}\alpha(\eta)
$$


### Sub Problem (c)

$$
\ell(\theta) = -\sum_{i = 1}^{m}log(p(y^{(i)}|x^{(i)};\theta)) \\
 = -\sum_{i = 1}^{m}(log(b(y)\, + \,(\eta^TT(y) \, - \,\alpha(\eta)))
$$

So the derivative is：
$$
\frac{\part \ell (\theta)}{\part \theta_j} = \sum_{i = 1}^{m}x_{j}^{(i)}\frac{\part \alpha(\theta^Tx)}{\part \theta_j} - x_j^{(i)}y^{(i)}
$$
The positive definiteness has been proved before, so we don't prove again.

## Problem 5

### Sub Problem (a)

#### problem i

Vectorize the $X,Y,\theta$，and we define the weight matrix：
$$
W_{ij} = \frac{1}{2}w^{(i)} \;(i = j)\\
W_{ij} = 0\;\;(i \neq j)
$$
So it can be easily proved.

#### problem ii

$$
\bigtriangledown_{\theta}J(\theta) = \bigtriangledown_{\theta}(\theta^TX^TWX\theta - y^TWX\theta - \theta^TX^TWy + y^TWy) \\
= \bigtriangledown_{\theta}(\theta^TX^TWX\theta - 2y^TWX\theta)\\
= 2X^TWX\theta - 2y^TWX = 0
$$

So the learning parameter vector $\theta$ is：
$$
\theta = (X^TWX)^{-1}y^TWX
$$

#### problem iii

$$
\frac{\part \ell(\theta)}{\part \theta_j} = \sum_{i = 1}^{m}-\frac{(y^{(i)} - \theta^Tx^{(i)})}{(\sigma^{(i)})^2}x_{j}^{(i)}
$$

So it is a weighted gradient descent, and the weight is about the variance $\sigma$

### Sub Problem (b)

```python
import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x,self.y = x,y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        pred = np.zeros(n)
        for i in range(m):
            Weight = np.diag(np.exp(-np.sum(self.x - x[i]) ** 2,axis=1) / (2 * (self.tau ** 2)))
            pred[i] = np.linalg.inv(self.x.T.dot(Weight).dot(self.x)).dot(self.x.T).dot(Weight).dot(self.y).T.dot(x[i])
        return pred
        # *** END CODE HERE ***

```



### Sub Problem (c)

```python
import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)
    MSE = []
    for tau in tau_values:
        model.tau = tau
        pred = model.predict(x_eval)
        squared_estimate = np.mean((pred - y_eval) ** 2)
        MSE.append(squared_estimate)
        
        plt.figure()
        plt.title('tau = {}'.format(tau))
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_eval, pred, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('p05c_tau_{}.png'.format(tau))

    tau_output = tau_values[np.argmin(MSE)]
    model.tau = tau_output
    pred = model.predict(x_test)
    np.savetxt(pred_path,pred)
    mse = np.mean((pred - y_eval) ** 2)


    # *** END CODE HERE ***

```

