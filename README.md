# reparam_autoencoder

This project implement the idea in paper "Auto-Encoding Variational Bayes"

Given a set of data points. We believe that there is an underlyding latent structure influence on our observations.

However, posterior distribution p(z|x) = p(x|z)p(z) / Z; where Z is intractable. If p(x|z) and p(z) not conjugate prior

We proposed easy computable q(z|x) and optimize the distance between q(z|x) and true posterior p(z|x) through autoencoder; 

When reconstruction image is similar to input image, we say that latent space is well-modeled
