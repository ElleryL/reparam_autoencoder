# reparam_autoencoder

This project implement the idea in paper "Auto-Encoding Variational Bayes"

So given any set of Bernoulli distributed data, this model will learn the underlyding latent structure of data where we propose the latent posterior dist as Gaussian

We use mnist binary image data set as a tester.

Future Plan:
(1) Generalized into categorical and continuous input data distribution
(2) It will incorporate reLAX estimator from grad_est to deal with discrete latent posterior dist 
