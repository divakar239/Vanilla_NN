# Vanilla_NN

### Objective:

  - In this file we will implement all the functions required to build a neural network.
  - Build a deep neural network with more than 1 hidden layer
  - Implement an easy to use neural network class
  
### Notation:
  - Superscript [l] denotes an entity associated with the l<sup>th</sup> layer of the neural network
  - Superscript (i) denotes an entity associated with the i<sup>th</sup> example
  - Lowerscript <i>i</i> denotes i<sup>th</sup> entry of a vector
  
  For example a<sub>i</sub><sup>[l]</sup> denotes the i<sup>th</sup> entry in the l<sup>th</sup> activation layer.

### Packages Required
- numpy
- matplotlib
- NNUtils.py (in the repo)

### Initialization 
This part will contain two helper functions that will initialize the parameters of the model. The first function will do so for a two layer model and the second function will do so for 'L' layers.

##### 2-Layer NN
The model's structure is as follows:
- LINEAR -> RELU -> LINEAR -> SIGMOID
- Weight matrices are randomly initialized by using np.random.rand(shape)*0.01
- The biases are initialized to 0 using np.zeros(shape)
- Fucntion signature is : initialize_parameters(n_x, n_h, n_y)

##### L-Layer NN
The model's structure is as follows:
- [LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID
- Weight matrices are randomly initialized by using np.random.rand(shape)*0.01
- The biases are initialized to 0 using np.zeros(shape)
- We will store n<sup>[l]</sup>, the number of units in different layers, in a variable layer_dims. 
- For example, the layer_dims for the "Planar Data classification model" from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit.

### Forward Propagation
This step builds on the following simple functions:
- LINEAR
- LINEAR -> ACTIVATION where the ACTIVATION will be either ReLU or sigmoid
- [LINEAR -> ReLU] x (L-1)->LINEAR->SIGMOID (entire model)
- The linear forward module (vectorized over all the examples) computes the following:
Z<sup>[l]</sup> = W<sup>[l]</sup>A<sup>[l-1]</sup> + b<sup>[l]</sup> where A<sup>[0]</sup> = X

##### Linear Activation Forward
-  **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$. We have provided you with the `sigmoid` function. This function returns **two** items: the activation value "`a`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call: 
``` python
A, activation_cache = sigmoid(Z)
```

- **ReLU**: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. We have provided you with the `relu` function. This function returns **two** items: the activation value "`A`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call:
``` python
A, activation_cache = relu(Z)
```
### Cost Function
Cross-entropy cost $J$, using the following formula: $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{7}$$

### Backward Propagation
Similar to forward propagation, you are going to build the backward propagation in three steps:
- LINEAR backward
- LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
- [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID backward (whole model)

##### Linear Backward
For layer $l$, the linear part is: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ (followed by an activation).

Suppose you have already calculated the derivative $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. You want to get $(dW^{[l]}, db^{[l]} dA^{[l-1]})$.

<img src="images/linearback_kiank.png" style="width:250px;height:300px;">
<caption><center> **Figure 4** </center></caption>

The three outputs $(dW^{[l]}, db^{[l]}, dA^{[l]})$ are computed using the input $dZ^{[l]}$.Here are the formulas :
$$ dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{8}$$
$$ db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{9}$$
$$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{10}$$

##### Linear Activation Backward
To implement `linear_activation_backward`, these two backward functions are required:
- **`sigmoid_backward`**: Implements the backward propagation for SIGMOID unit. You can call it as follows:

```python
dZ = sigmoid_backward(dA, activation_cache)
```

- **`relu_backward`**: Implements the backward propagation for RELU unit. You can call it as follows:

```python
dZ = relu_backward(dA, activation_cache)
```

If $g(.)$ is the activation function, 
`sigmoid_backward` and `relu_backward` compute $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}) \tag{11}$$.  

##### L-Model Backward
**Initializing backpropagation**:
To backpropagate through this network, the output is, 
$A^{[L]} = \sigma(Z^{[L]})$. Your code thus needs to compute `dAL` $= \frac{\partial \mathcal{L}}{\partial A^{[L]}}$.
To do so, this formula is used:
```python
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
```

-This post-activation gradient `dAL` is used to keep going backward. 
- `dAL` is fed into the LINEAR->SIGMOID backward function you implemented (which will use the cached values stored by the L_model_forward function). 
- After that,  a `for` loop is used to iterate through all the other layers using the LINEAR->RELU backward function. 
-  Each dA, dW, and db is stored in the grads dictionary. 

$$grads["dW" + str(l)] = dW^{[l]}\tag{15} $$

For example, for $l=3$ this would store $dW^{[l]}$ in `grads["dW3"]`.

### Updating Parameters
The parameters are updated as follows:

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$

where $\alpha$ is the learning rate. The updated parameters are stored in the parameters dictionary. 
