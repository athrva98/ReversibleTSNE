# ReversibleTSNE
This is an extension of sklearn's TSNEwhich allows the calculation of an inverse transform from the embedding to the original data.

## Calculation of the inverse transform
It is known that there is no analytic way of computing an inverse transform for TSNE, the cost function in TSNE is non-convex, and an inverse mapping is not always defined.
To that end, we can try to approximate the inverse mapping. This can be done using Machine Learning, or via Global Optimization. If the number of samples is too small, I prefer the second approach (although computationally expensive). In this repository I provide support for the calculating of an inverse transformation using optimizers in scipy's minimize, differential_evolution and dual_annealing submodules.
