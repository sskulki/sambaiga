---
layout: post
title: Learning HMM parameters for Continous Density Models
date: 2017-06-12 11:12:00-0400
categories: [ML, HMM]
thumbnail: img/post/ghmm.jpg
summary: >
  This post review training methods for learning standard HMM parameters with gaussian observation.

mathjax: true
---


In the previous post we considered a scenario in which observation sequences $$Y$$ are discrete symbols. However, for many practical problems the observation symbols are continous vectors. As the results the contious probability desnsity function (pdfs) are used to model the space of the observation signal associated with each state. Most commonly used emission distribution are gaussian distribution and the gausian mixture models. 


### Gaussian Distribution and the Gausian Mixture Models

It is popular to represent the randomness of continuous-valued  using the multivariate Gaussian distribution. A vector-valued random variable $$\mathbf{x}$$ is said to have a multivariate normal (or Gaussian) distribution with mean $$\mu=\mathop{\mathbf{E[x]}}$$ and covariance matrix $$\Sigma=\mathbf{cov[x]}$$ if: 

$$
 P(\mathbf{x}; \mu, \Sigma) = \mathcal{N(\mathbf{x} \mid \mu, \Sigma)}=\frac{1}{(2\pi)^{D/2} |\Sigma|^\frac{1}{2}}\quad\exp\Big(-\frac{1}{2}[\mathbf{x} - \mu] \Sigma^{-1}[\mathbf{x} - \mu]^\mathsf{T} \Big) 
$$


where $$D$$ is the dimensionality of $$\mathbf{x}$$. The $$\mu$$ represents the location where samples are most likely to be generated and the $$\Sigma$$ indicates the level to which two variables vary together.

However, a single Gaussian distribution is insufficient to represent the state-dependent observation space for an HMM state $$s_t=i$$ because there are large amounts of training data collected from various appliance instances with different modes, distortions, background noises, etc which are used to train the parameters of individual HMM states. In this case, a Gaussian mixture model (GMM) is adopted to represent the state-dependent observation space.

A mixture model is a probabilistic model for density estimation using a mixture distribution and can be regarded as a type of unsupervised learning or clustering. They provide a method of describing more complex propability distributions, by combining several probability distributions. A multivariate Gaussian mixture distribution is given by the following equation:

$$ 
P(\mathbf{x}) = \displaystyle\sum_{k=1}^{K}\omega_k \mathcal{N(\mathbf{x} \mid \mu_k, \Sigma_k)}   
$$

The parameters $$\omega_k$$ are called mixing coefficients, which must fulfill  

$$
\displaystyle\sum_{k=1}^{K}\omega_k =1 
$$

and given $$\mathcal{N(\mathbf{x} \mid \mu_k, \Sigma_k)} \geq 0$$ and $$P(\mathbf{x}) \geq 0$$ we also have that 
$$0\leq \omega_k \geq 1$$. Each Gaussian density $$\mathcal{N(\mathbf{x} \mid \mu_k, \Sigma_k)}$$ is
called a component of the mixture and has its own mean $$\mu_k$$   and covariance $$\Sigma_k$$.


### HMM with gaussian emission distribution 

If the observations are continuous, it is common for the emission probabilities to be a conditional
Gaussian such that:

$$
P(\mathbb{y_t} \mid s_t =i) = \mathcal{N(\mathbf{y_t} \mid \mu_i, \Sigma_i)} 
$$

 where $$\mu_i$$ and $$\Sigma_i$$ are mean vector and covariance matrix associated with state $$i$$.

The re-estimation formula for the mean vector and covariance matrix of a state gausian pdf can be derived as:


$$
\begin{aligned}
 \hat{\mu}_i & =\frac{\displaystyle\sum_{t=1}^{T}\gamma_t(i)\mathbb{y(t)}}{\displaystyle\sum_{t=1}^{T}\gamma _t(i)}\\
 \hat{\Sigma}_i & =\frac{\displaystyle\sum_{t=1}^{T}\gamma_t(i) [\mathbf{y(t)}-\hat{\mu}_i]\cdot[\mathbf{y(t)}-\hat{\mu}_i]^T}{\displaystyle\sum_{t=1}^{T}\gamma_t(i)}
\end{aligned}
$$


### HMMs with Gaussian Mixture Model

In HMMs with gaussian mixture pdf, the emission probabilities is given by 

$$
 P(\mathbb{y_t} \mid s_t =i) = \displaystyle\sum_{k=1}^{M} \omega\_{ik}\mathcal{N(\mathbb{y_t} \mid \mu_{ik}, \Sigma_{ik})}  
$$
  where $$\omega_{ik}$$ is the prior probability of the  $$k^{th}$$ component of the mixture.

The posterior probability of state $$s_t=i$$ at time $$t$$ and state $$s_{t+1}=j$$ at time $$t+1$$ given the model $$\lambda$$ and the observation sequence $$Y$$ is

$$
\begin{aligned}
 \gamma_t(i,j)& =P(s_t=i, s_{t+1}=j \mid Y, \lambda) \\
 & = \frac{\alpha_t(i)a_{ij}\Big[ \displaystyle\sum_{k=1}^{M} \omega_{ik}\mathcal{N(\mathbf{y_t} \mid \mu_{ik}, \Sigma_{ik})} \Big]\beta_{t+1}(j)}{\displaystyle\sum_{i=1}^{N}\alpha_T(i)}
\end{aligned}
$$

 and the posterior probability of state $$s_t=i$$ at time $$t$$ given the model $$\lambda$$ and observation $$Y$$ is 

$$
\gamma_t(i) =\frac{\alpha_t(i)\beta_t(i)}{\displaystyle\sum_{i=1}^{N}\alpha _T(i)}
$$

Let define the joint posterior probability of the state $$s_i$$ and the $$k^{th}$$ gaussian component pdf of state $$i$$ at time $$t$$

$$
\begin{aligned}
\xi(i,k) &= P(S_t=s_i, m(t)=k \mid Y, \lambda) \\
 &=\frac{\displaystyle\sum_{j=1}^{N} \alpha_t(j) a_{ij} \omega_{ik}\mathcal{N(\mathbf{y_t} \mid \mu_{ik}, \Sigma_{ik})}\beta_{t+1}(j)}{\displaystyle\sum_{i=1}^{N}\alpha _T(i)} 
\end{aligned}
$$ 

The re-estimation formula for the mixture coefficeints, the mean vectors and the covariance matrices of the state mixture gausian pdf as

$$
\begin{aligned}
 \hat{\omega}_{ik} &= \frac{\displaystyle \sum_{t=1}^{T} \xi_t(i,k)}{\displaystyle\sum\_{t=0}^{T}\gamma_t(i)} \\
\hat{\mu}_{ik} &= \frac{\displaystyle\sum\ _{t=1}^{T}\xi\ _t(i,k)\mathbf{y_t}}{\displaystyle\sum_{t=1}^{T}\xi_t(i,k)} \\
\hat{\Sigma}_{ik}&=\frac{\displaystyle\sum_{t=1}^{T}\xi_t(i,k)[\mathbf{y_t}-\hat{\mu}_{ik}]\cdot[\mathbf{y_t}-\hat{\mu}_{ik}]^T}{\displaystyle\sum_{t=1}^{T}\xi_t(i,k)}
\end{aligned}
$$

### Limitation of Baum–Welch algorithm

When applying Baum–Welch algorithm  in real  data, we need to consider some heuristics in the ML EM algorithm.

1. How to provide initial parameters values. This is always an important question, and it is usually resolved by using a simple algorithm (e.g., K-means clustering or random initialization).
2. How to avoid unstability in the parameter estimation (especially covariance parameter estimation) due to data sparseness. For examle some mixture components or hidden states cannot have sufficient data assigned in the Viterbi or forward–backward algorithm. This can be heuristically avoided by setting a threshold to update these parameters, or setting minimum threshold values for covariance parameters.

The above two problem can be solved by the Bayesian approaches.

### References
1. Saeed V. Vaseghi, Advanced Digital Signal Processing and Noise Reduction. John Wiley & Sons, 2008.
2. Kevin P. Murphy, Machine Learning: A Probabilistic Perspective. The MIT Press Cambridge, Massachusetts, 2012.
