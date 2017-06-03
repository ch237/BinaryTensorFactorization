# BinaryTensorFactorization

This is a matlab implementation of scalable binary tensor factorization model, which contains both batch and online inferences.

* run demoBTFsmall.m for small binary tensors
* run demo_gibbs_large.m for large tensor data: including facebook and duke scholar datasets
* run demojointfactorization.m for joint matrix (side information) and tensor factorization, the code is used for cold start problem in which we want to infer the tensor entries that are completely missing for some entities

The code is related to our following two publications:

* C Hu, P Rai, L Carin. Zero-Truncated Poisson Tensor Factorization for Massive Binary Tensors, UAI 2015, Amsterdam, The Netherlands.
* C Hu, P Rai, C Chen, M Harding, L Carin. Scalable Bayesian Non-Negative Tensor Factorization for Massive Count Data, ECML-PKDD 2015, Porto, Portugal.


