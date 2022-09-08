# CS229 ps2-solution

## Problem 1

### Sub Problem (a)

Converges in dataset A , NOT converge in dataset B.

### Sub Problem (b)

Dataset A is NOT linear separable, but dataset B is linear separable. It's shown by the output graph by the program.

**Why does NOT linear regression converge on a linear separable dataset?**

As the dataset is linear separable, we can assume that the separate line is $x=0$ , otherwise we can rotate the space to make the separate line to be $x = 0$

Our loss function is :
$$
\ell(\theta) = \sum_{i = 1}^{m}y^{(i)}log(g(\theta^Tx)) + (1 - y^{(i)})log(1 - g(\theta^Tx))
$$
In this specific separable case, we can divide the data into only two kinds of types:

* $y^{(i)} = 1$ and $x^{(i)} > 0$
* $y^{(i)} = 0$ and $x^{(i)} < 0$

For the first type, it leads $log(g(\theta^Tx))$ to zero (because the derivative is larger than zero);

For the second type, it leads $log(1 - g(\theta^Tx))$ to zero(because the derivative is less than zero).

So in both cases, the loss function is NOT converging.

Conversely, when the dataset is not linear separable, we got four different types:

* $y^{(i)} = 1$ and $x^{(i)} > 0$
* $y^{(i)} = 1$ and $x^{(i)} < 0$
* $y^{(i)} = 0$ and $x^{(i)} > 0$
* $y^{(i)} = 0$ and $x^{(i)} < 0$

So its derivative will keep changing and the loss function must get a minimum point.

### Sub Problem (c)

#### i

**Do NOT work.**  Because changing the learning rate doesn't change the fact that the the logistic regression doesn't converges on a linear separable dataset.

#### ii

Yes, because as the time goes, the error of $\theta$ will be limited as we can tolerance.

#### iii

**Do NOT work.** Linear scaling is the same way with multiple a real number to $\theta$, doesn't change the properties of the model and the dataset.

#### iv

When adding a $L_2$ regularization, the derivative of loss function becomes: (Set the const of $L_2$ regularization is 1/2)
$$
\frac{\partial J}{\partial \theta} = x_j^{(i)}(y^{(i)} - g(\theta^Tx)) + \theta_j
$$
So the update of $\theta_j$ is:
$$
\theta_j = (1 - \alpha)\theta_j - \alpha x_j^{(i)}(y^{(i)} - g(\theta^Tx))
$$
It can prevent the arbitrary scaling. So it does work.

**We can adjust the weight of the regularization term to promise the scaling is not arbitrary.**

#### v

Yes, adding noise term could make the model converges. Just have to make sure to keep the $\theta$ not arbitrary scaling.

### Sub Problem (d)



## Problem 2

This problem tends to make us understand more that the output of logistic regression is the empirical probability of the training set (**NOT only because the sigmoid function output a value between 0 and 1**)

### Sub Problem (a)

As the problem says, the parameter $\theta$ is maximum likelihood parameter, so we calculate $\theta$ ：
$$
\begin{align*}
\ell(\theta) &= \sum_{i = 1}^{m}y^{(i)}log(h(x^{(i)})) + (1 - y^{(i)})log(1 - h(x^{(i)})) \\
&= \sum_{i = 1}^{m}-y^{(i)}log(1 + e^{-\theta^Tx}) + (1 - y^{(i)})log(\frac{e^{-\theta^Tx}}{1 + e^{-\theta^Tx}}) \\
&= \sum_{i = 1}^{m}-y^{(i)}log(1 + e^{-\theta^Tx}) + (1 - y^{(i)})(-\theta^Tx) - (1 - y^{(i)})log(1 + e^{-\theta^Tx})\\
&= \sum_{i = 1}^{m}(1 - y^{(i)})(-\theta^Tx) + log(1 + e^{-\theta^Tx}) \\
\frac{\partial \ell}{\partial \theta_j} &= \sum_{i = 1}^{m}(1 - y^{(i)})(-x_{j}^{(i)}) + (1 - g(\theta^Tx))x_{j}^{(i)}\\
&= \sum_{i = 1}^{m}x_{j}^{(i)}(y^{(i)} - g(\theta^Tx))
\end{align*}
$$
Let j = 0 (because the problem sets $x_{0}^{(i)}=1$), then we got:
$$
\sum_{i = 1}^{m}(y^{(i)} - g(\theta^Tx)) = 0
$$
So we proof the problem.

### Sub Problem (b)

The conclusion is **NOT TRUE**.

We set (a,b) equals to (0.5,1) , so we got :
$$
\sum_{i \in I_{a,b}}I(y^{(i)} = 1) = |\{i \in I_{a,b}\}|
$$
But 
$$
\sum_{i \in I_{a,b}}P(y = 1|x;\theta) < |\{i \in I_{a,b}\}|
$$
So the formula can't set up.

### Sub Problem (c)

After adding $L_2$ regularization, calculating the maximum likelihood,we got:
$$
\sum_{i = 1}^{m}(g(\theta^Tx) - y^{(i)})x_{j}^{(i)} + \lambda\theta = 0
$$
that means:
$$
\sum_{i = 1}^{m}P(y = 1|x;\theta) + \lambda\theta = \sum_{i = 1}^{m}I\{y^{(i)} = 1\}
$$
So the formula can't set up.

## Problem 3

This problem lets us explore the connection between MAP(maximum a posteriori estimation) and MLE(maximum likelihood estimation), and how to choose prior distribution over $\theta$.

### Sub Problem (a)

$$
\begin{align*}
\theta_{MAP} &= arg\,max_\theta\; p(\theta|x,y)\\
p(\theta|x,y) &= \frac{p(\theta,x,y)}{p(x,y)} = \frac{p(y|x,\theta)p(\theta,x)}{p(x,y)} = \frac{p(y|x,\theta)p(\theta|x)p(x)}{p(x,y)} \\
&= p(y|x,\theta)p(\theta|x) = p(y|x,\theta)p(\theta)
\end{align*}
$$

So we finish the prove.

### Sub Problem (b)

From sub problem (a), we know:
$$
\theta_{MAP} = arg\,max_{\theta}p(y|x,\theta)p(\theta)
$$
So,it is the same with:
$$
\theta_{MAP} = arg\,max_{\theta}\;(log\;p(y|x,\theta) + log\;p(\theta))
$$
Because $\theta$ ~ $\mathcal{N}(0,\eta^2I)$, so we have:
$$
\begin{align*}
p(\theta) &= \frac{1}{(2\pi)^\frac{n}{2}\eta^n}exp\{-\frac{||\theta||^2}{2\eta^2}\} \\
log\,p(\theta) &= -\frac{n}{2}log\,(2\pi) - nlog(\eta) - \frac{||\theta||^2}{2\eta^2}
\end{align*}
$$
So, we have:
$$
\begin{align*}
\theta_{MAP}  &= arg\,max_{\theta}\;(\,log\,p(y|x,\theta)\,- \frac{1}{2\eta^2}||\theta||^2) \\
 &= arg\,min_{\theta}\;(-log\,p(y|x,\theta) + \frac{1}{2\eta^2}||\theta||^2)
\end{align*}
$$
We finish the prove.

And $\lambda = \frac{1}{2\eta^2}$

### Sub Problem (c)

$$
p(y|X;\theta) = \frac{1}{(2\pi)^{\frac{m}{2}}\sigma^m}exp\{-\frac{1}{2\sigma^2}||X\theta - y ||^2\}
$$

The same with sub problem (b), we have:
$$
\begin{align*}
log\,p(y|X,\theta) &= -\frac{m}{2}log(2\pi) - mlog\sigma - \frac{1}{2\sigma^2}||X\theta - y||^2\\
\theta_{MAP} &= arg\,min_{\theta}(-log\,p(y|x,\theta) + \frac{1}{2\eta^2}||\theta||^2) \\
&= arg\,min_{\theta}(\frac{1}{2\sigma^2}||X\theta - y||^2 + \frac{1}{2\eta^2}||\theta||^2)\\
J &= arg\,min_{\theta}(\frac{1}{2\sigma^2}||X\theta - y||^2 + \frac{1}{2\eta^2}||\theta||^2)
\end{align*}
$$
Let derivative be zero:
$$
\frac{\partial J}{\partial \theta} = \frac{1}{\sigma^2}(X^TX\theta - X^Ty) + \frac{1}{\eta^2}\theta = 0
$$
So:
$$
\theta_{MAP} = arg\,min_{\theta}J = (X^TX + \frac{\sigma^2}{\eta^2}I)^{-1}X^Ty
$$


### Sub Problem (d)

 The derivation is similar with the question before, just change the distribution of $\theta$ :
$$
\begin{align*}
p(\theta) &= \frac{1}{(2b)^n}exp\{-\frac{1}{b}||\theta||_1\} \\
\theta_{MAP} &= arg\,min\,\frac{1}{2\sigma^2}||X\theta - y||^2 + \frac{1}{b}||\theta||_1 \\
J(\theta) &= ||X\theta-y||^2 + \gamma||\theta||_1
\end{align*}
$$
So, we got:
$$
\gamma = \frac{2\sigma^2}{b}
$$


## Problem 4

### Sub Problem (a)

$K(x,z) = K_1(x,z) + K_2(x,z)$ **is a kernel.**

 Because the sum of two positive semi-define matrices is still a positive semi-define matrix. It can be shown by the definition of the positive semi-define matrix

### Sub Problem (b)

$K(x,z) = K_1(x,z) - K_2(x,z)$ **is NOT a kernel.**

Consider this example:
$$
\begin{bmatrix}
I_r & O	\\
O & O 
\end{bmatrix}
-
\begin{bmatrix}
I_{r + 1} & O \\
O & O
\end{bmatrix}
=
\begin{bmatrix}
-I_1 & O \\
O & O
\end{bmatrix}
$$
the difference of these two positive semi-define matrices is a negative semi-define matrices, from the Mercer's theory we know, it can't be a kernel.

### Sub Problem (c)

$K(x,z) = aK_1(x,z)$ **is a kernel.**

Obviously, a positive semi-define matrix multiply a positive number(it's the same with multiply a identify matrix times a constant) is still a positive semi-define matrix. Because it has the same standard matrix with the original matrix.

### Sub Problem (d)

$K(x,z) = -aK_1(x,z)$ **is NOT a kernel.**

Its standard matrix is the opposite with the original matrix's standard matrix.

### Sub Problem (e)

 $K(x,z) = K_1(x,z)K_2(x,z)$ **is a kernel.**

the $K_1(x,z)$ and $K_2(x,z)$ return  none negative values, so the $K(x,z)$ returns a none negative value.

### Sub Problem (f)

Calculate the new kernel can prove:
$$
\begin{align*}
z^TKz &= \sum_i\sum_jz_iK_{ij}z_j \\
 &= \sum_i\sum_jz_if(x^{(i)})f(x^{(j)})z_j	\\
  &= \sum_i(z_if(x^{(i)}))^2\\
  &\geq 0
\end{align*}
$$


### Sub Problem (g)

$K(x,z) = K_3(\phi(x),\phi(z))$ **is a kernel.**

No matter the vector mapped by $\phi$ is, the Gram Matrix of $K_3$ is a positive semi-define matrix. So it is a kernel.

### Sub Problem (h)

$K(x,z) = p(K_1(x,z))$ **is a kernel.**

Because $K_1$ is a kernel and the mapping function $p$ is a positive coefficient, so the value of $K_1(x,z)$ is none negative, so the value of $K(x,z)$ is none negative.

## Problem 5

### Sub Problem (a)

#### i

 Observing the $h_{\theta}$ , we know that every update of $\theta$ is a linear combination of $\phi(x^{(i)})$, so we have bunch of real parameters $\beta$ to represent: $\theta = \sum_{i = 1}^{n}\beta_i\phi(x^{(i)})$

#### ii

From the question above, we know that every update makes $\theta$ a combination of $\phi(x^{(i)})$, so when predicting the next data point, the computation becomes 
$$
\begin{align*}
g({\theta^{(i)}}^T\phi(x^{(i + 1)})) &= g((\sum_{k = 1}^i\beta_k\phi(x^{(k)}))^T\phi(x^{(i + 1)})) \\
&= \sum_{k = 1}^i\beta_k(\phi(x^{(k)})\phi(x^{(i + 1)})) \\
&= \sum_{k = 1}^i\beta_kK(x^{(k)},x^{(i +1)})
\end{align*}
$$
So, we can use the kernel function to predict the unknown point in the data set.

#### iii

We can update $\beta$ by using the same way as updating $\theta$,:
$$
\beta_i = \alpha(y^{(i)} - g({\theta^{(i)}}^T\phi(x^{(i)})))
$$
As i loops.

### Sub Problem (b)

```python
import math

import matplotlib.pyplot as plt
import numpy as np

import util


def initial_state():
    """Return the initial state for the perceptron.

    This function computes and then returns the initial state of the perceptron.
    Feel free to use any data type (dicts, lists, tuples, or custom classes) to
    contain the state of the perceptron.

    """

    # *** START CODE HERE ***
    return []
    # *** END CODE HERE ***


def predict(state, kernel, x_i):
    """Peform a prediction on a given instance x_i given the current state
    and the kernel.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns
            the result of a kernel
        x_i: A vector containing the features for a single instance
    
    Returns:
        Returns the prediction (i.e 0 or 1)
    """
    # *** START CODE HERE ***
    if sum(beta * kernel(x,x_i) for beta,x in state) < 0:
        return 0
    return 1
    # *** END CODE HERE ***


def update_state(state, kernel, learning_rate, x_i, y_i):
    """Updates the state of the perceptron.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    # *** START CODE HERE ***
    beta_i = learning_rate * (y_i - sign(sum(beta * kernel(x,x_i) for beta,x in state)))
    state.append((beta_i,x_i))
    # *** END CODE HERE ***


def sign(a):
    """Gets the sign of a scalar input."""
    if a >= 0:
        return 1
    else:
        return 0


def dot_kernel(a, b):
    """An implementation of a dot product kernel.

    Args:
        a: A vector
        b: A vector
    """
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    """An implementation of the radial basis function kernel.

    Args:
        a: A vector
        b: A vector
        sigma: The radius of the kernel
    """
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)


def train_perceptron(kernel_name, kernel, learning_rate):
    """Train a perceptron with the given kernel.

    This function trains a perceptron with a given kernel and then
    uses that perceptron to make predictions.
    The output predictions are saved to src/output/p05_{kernel_name}_predictions.txt.
    The output plots are saved to src/output_{kernel_name}_output.pdf.

    Args:
        kernel_name: The name of the kernel.
        kernel: The kernel function.
        learning_rate: The learning rate for training.
    """
    train_x, train_y = util.load_csv('../data/ds5_train.csv')

    state = initial_state()

    for x_i, y_i in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, x_i, y_i)

    test_x, test_y = util.load_csv('../data/ds5_train.csv')

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda a: predict(state, kernel, a))
    util.plot_points(test_x, test_y)
    plt.savefig('./output/p05_{}_output.pdf'.format(kernel_name))

    predict_y = [predict(state, kernel, test_x[i, :]) for i in range(test_y.shape[0])]

    np.savetxt('./output/p05_{}_predictions'.format(kernel_name), predict_y)


def main():
    train_perceptron('dot', dot_kernel, 0.5)
    train_perceptron('rbf', rbf_kernel, 0.5)


if __name__ == "__main__":
    main()

```

### Sub Problem (c)

Dot kernel performance badly, because it's mapping $\phi(x) = x$, doesn't change the fact that the dataset is linear separable.

