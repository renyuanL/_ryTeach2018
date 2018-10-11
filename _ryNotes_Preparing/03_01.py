
>>> help(iris)

# Help on Bunch in module sklearn.utils object:

class Bunch(builtins.dict)
 |  Container object for datasets
 |  
 |  Dictionary-like object that exposes its keys as attributes.
 |  
 |  >>> b = Bunch(a=1, b=2)
 |  >>> b['b']
 |  2
 |  >>> b.b
 |  2
 |  >>> b.a = 3
 |  >>> b['a']
 |  3
 |  >>> b.c = 6
 |  >>> b['c']
 |  6
 |  
 |  Method resolution order:
 |      Bunch
 |      builtins.dict
 |      builtins.object
 
 >>> help(train_test_split)
 
 # Help on function train_test_split in module sklearn.model_selection._split:

def train_test_split(*arrays, **options)
  | Split arrays or matrices into random train and test subsets
  |

>>> help(np.bincount)
# Help on built-in function bincount in module numpy.core.multiarray:

bincount(...)
    bincount(x, weights=None, minlength=0)
    
    Count number of occurrences of each value in array of non-negative ints.
    
>>> help(StandardScaler)

class StandardScaler(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin)
 |  Standardize features by removing the mean and scaling to unit variance
 |  
 |  Centering and scaling happen independently on each feature by computing
 |  the relevant statistics on the samples in the training set. Mean and
 |  standard deviation are then stored to be used on later data using the
 |  `transform` method.
 |  
 |  Standardization of a dataset is a common requirement for many
 |  machine learning estimators: they might behave badly if the
 |  individual feature do not more or less look like standard normally
 |  distributed data (e.g. Gaussian with 0 mean and unit variance).
 
>>> help(Perceptron)

Help on class Perceptron in module sklearn.linear_model.perceptron:

class Perceptron(sklearn.linear_model.stochastic_gradient.BaseSGDClassifier)
 |  Perceptron
 |  
 |  Read more in the :ref:`User Guide <perceptron>`.

>>> help(np.vstack)
Help on function vstack in module numpy.core.shape_base:

vstack(tup)
    Stack arrays in sequence vertically (row wise).
    
    Take a sequence of arrays and stack them vertically to make a single
    array. Rebuild arrays divided by `vsplit`.

>>> help(np.stack)
Help on function stack in module numpy.core.shape_base:

stack(arrays, axis=0)
    Join a sequence of arrays along a new axis.
    
    The `axis` parameter specifies the index of the new axis in the dimensions
    of the result. For example, if ``axis=0`` it will be the first dimension
    and if ``axis=-1`` it will be the last dimension.
