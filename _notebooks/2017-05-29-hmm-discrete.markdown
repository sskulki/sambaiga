---
layout: post
title: Learning HMM parameters with Discrete Observation Models
date: 2017-05-29 11:12:00-0400
categories: [ML, HMM]
thumbnail: img/post/dhmm.jpg
summary: >
  This post review E-M method for training standard HMM models with discrete observation. 

mathjax: true
---

In Previous post we discussed the basic of HMM modeling given model parameters $$\lambda$$ and  compute the likelihood values etc, efficiently based on the forward, backward, and Viterbi algorithms. In the like manner, we can efficiently train the HMM to obtain the model parameter $$\hat{\lambda}$$ from data. In this post we will discuss different methods for training HMM models. 

This is the solution to Problem 3 which involve determining a method to learn model parameters $$\hat{\lambda}$$ given the sequence of observation variables $$Y$$. Given the observation sequences $$Y$$ as training data, there is no optimal way of estimating the model parametrs. However, using iterative procedure we can choose $$\hat{\lambda} = (\hat{A},\hat{B},\hat{\pi})$$ such that $$P(Y \mid \lambda)$$ is locally maximized.The most common produre which has been employed to his problem is the **Baum-Welch** method.


### Baum-Welch Methods 

This method assume an initial model parameters $$\lambda$$ which should be adjusted so as to increase $$P(Y \mid \lambda)$$. The initial parametrs can be constructed in any way or employ the first five procedure of the [Segmental K-means algorithm](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/segmental%20k-means%20algorithm.pdf). The optimazation criteria is called the **maximum likelihood criteria**.The function $$P(Y \mid \lambda)$$ is called the **likelihood function**.



### The E-M Auxilliary Function

Let $$\lambda$$ represent the current model and $$\hat{\lambda}$$ represent the candidate models. The learning objective is to make: $$P(Y \mid \hat{\lambda}) \geq P(Y \mid \lambda)$$ which is equivalently to $$ \log[P(Y \mid \hat{\lambda})] \geq \log [P(Y\mid \lambda)]$$

Let also define the auxilliary function $$Q(\hat{\lambda}\mid \lambda)$$ such that:



$$
\begin{aligned}
Q(\hat{\lambda}\mid \lambda) & = \mathbb{E}\Big[\log P(Y,S \mid \hat{\lambda})\mid Y, \lambda \Big] \\
                            & = \sum_s P(S \mid Y, \lambda)\cdot \log [P(Y,S\mid \hat{\lambda})]
\end{aligned}
$$


The Maximum Likehood Estimation (MLE) of the model parameter $$\lambda$$ for complete data $$Y$$ and hidden state $$S$$ is; 

$$ 
\hat{\lambda} = \arg\max _{\lambda} \sum_s P(Y, S \mid \lambda) 
$$

However due to the presence of several stochatsic constraints it turns out to be easier to mximize uxilliary function $$Q(\hat{\lambda}\mid \lambda)$$ rather than directly maximize $$P(Y\mid \hat{\lambda})$$. Thus the MLE of the model parameter $$\lambda$$ for complete data $$Y$$ and hidden state $$S$$ become:

$$
\hat{\lambda} = \arg\max _{\lambda} Q(\hat{\lambda}\mid\lambda)
$$

It can be shown that the parameter estimated by the EM procedure, $$Q(\hat{\lambda}\mid \lambda)$$, always increases the likelihood value. You may concert [reference 2](https://books.google.co.tz/books/about/Bayesian_Speech_and_Language_Processing.html?id=rEzzCQAAQBAJ&printsec=frontcover&source=kp_read_button&redir_esc=y#v=onepage&q&f=false) chapter 3 for details on the prove.  



### Expectation step

To find ML estimates of HMM parameters, we first expand the auxiliary function rewrite it by substituting the joint distribution of complete data likelihood.

$$
\begin{aligned}
Q(\hat{\lambda} \mid \lambda) & = \mathbf{E}\Big[\log P(Y,S\mid\hat{\lambda})\mid Y, \lambda \Big] \\
& = \sum_s P(S\mid Y, \lambda)\cdot \log [P(Y,S\mid\hat{\lambda})] \\
& = \sum_s P(S\mid Y, \lambda)\cdot \Big[\log \hat{\pi}_1 + \log \hat{b}_1(y_1) + \sum _{t=2}^T\big( \log \hat{a} _{ij} + \log \hat{b}_i({y_t})\big)\Big] 
\end{aligned}
$$

We have three term to solve: 
- The initial probability $$\hat{\pi}$$ , 
- State transition probability $$\hat{A} = \hat{a}_{ij}$$ and 
- Emission probability $$\hat{B} = \hat{b}_i(y_t)$$. 

Let first define important parameters that we will use. For $$t = 1,2...T$$, $$1\leq i \geq N$$ and $$1\leq j \geq N$$, we define:

$$
\xi_t(i,j)=P(s_t=i, s_{t+1}=j \mid Y, \lambda)
$$ 

an expected transition probability from $$s_t=i$$ to , $$s_{t+1}=j$$. The probability of being in state $$s_i$$ at time $$t$$ and state $$s_j$$ at time $$t+1$$ given the model $$\lambda$$ and observation sequences $$Y$$.

 $$\xi_t(i,j)$$ can be written in terms of forward $$\alpha_t(i)$$ and backward $$\beta_{t+1}(j)$$ variables as:

$$
\begin{aligned}   
\xi_t(i,j) &= \frac{\alpha_t(i)a_{ij}b_i(y_{t+1})\beta_{t+1}(j)}{P(Y \mid \lambda)} \\ 
          &= \frac{\alpha_t(i)a_{ij}b_i(y_{t+1})\beta_{t+1}(i)}{\displaystyle \sum_{i=1}^{N}\displaystyle \sum_{j=1}^{N}\alpha_t(i)a_{ij}b_j(y_{t+1})\beta_{t+1}(j)}
\end{aligned}
$$

where the numerator term is just $$P(S_t=s_i, S_{t+1}=s_j \mid Y, \lambda)$$  and the division by $$P(Y \mid \lambda)$$ gives the desire probability measures. 

We have previosly difined $$
\gamma_t(i) =  \frac{\alpha_t(i)\beta_t(i)}{P(Y \mid \lambda)}
$$ as the probability of being in state $$s_i$$ at time $$t$$ given the observation sequence and model parameter. $$\gamma_t(i)$$ relate to $$\xi_t(i,j)$$ as follows: 

$$
 \gamma_t(i) = \displaystyle\sum_{j=1}^{N}\xi_t(i,j) 
$$ 

It follows that:

 $$\displaystyle\sum_{t=1}^{T-1}\gamma_t(i)=$$ Expected number of transitions from state $$i$$ .

 $$\displaystyle\sum_{t=1}^{T-1}\xi_t(i,j)=$$ Expected number of transitions from state $$i$$ to state $$j$$.

We provide the solution for each term. Considering the first term  $$Q(\hat{\pi} \mid \pi)$$ we define the following auxiliary function for $$\pi _i$$  as:

$$
  Q(\hat{\pi}\mid \pi) = \sum_s P(S\mid Y, \lambda)\cdot \log \hat{\pi}_{s_1}
$$

Since $$\hat{\pi}_{s_1}$$ only depends on $$s_1$$, it clear that:

$$
P(S\mid Y, \lambda) = P(s_1\mid Y, \lambda)
$$

Therefore $$Q(\hat{\pi}\mid pi)$$  can be rewritten as:

$$
\begin{aligned}
Q(\hat{\pi}\mid \pi) &= \sum_{s_1}P(s_1 \mid Y, \lambda)\cdot \log \hat{\pi}_{s_1} \\
                     &= \sum_{i=1}^N P(s_1=i \mid Y, \lambda)\cdot \log \hat{\pi}_{i} \\
                     & = \sum_{i=1}^N \gamma_t(i) \log \hat{\pi}_{i}
\end{aligned}
$$

Next, we focus on the second term $$Q(\hat{A}\mid A)$$

$$
Q(\hat{A}\mid A) = \sum_s P(S\mid Y, \lambda) \cdot \sum _{t=2}^T  \log \hat{a}_{s_t,s_{t+1}}
$$

Similar to $$Q(\hat{\pi}\mid \pi)$$ , we obtain
$$
P(S\mid Y, \lambda) = P(s_1\mid Y, \lambda) = P(s_t,s_{t+1}\mid Y, \lambda)
$$

Therefore

$$
\begin{aligned}
Q(\hat{A}\mid A) & =  \sum _{t=1}^{T-1}  \sum_s P(s_t,s_{t+1}\mid Y, \lambda) \log \hat{a}_{s_t,s_{t+1}} \\
 & = \sum _{t=1}^{T-1} \sum_{i=1}^N \sum_{j=1}^N P(s_t=i,s_{t+1}=j\mid Y, \lambda) \log \hat{a}_{ij} \\
& = \sum _{t=1}^{T-1} \sum_{i=1}^N \sum_{j=1}^N \xi_t(i,j) \log \hat{a}_{ij}
\end{aligned}
$$

Finally, we focus on the last term $$Q(\hat{B}\mid B)$$

$$
Q(\hat{B}\mid B) = \sum_s P(S\mid Y, \lambda)\cdot \sum _{t=1}^T  \log \hat{b}_{i}(y_t)
$$

Similary $$P(S\mid Y, \lambda) = P(s_t = i\mid Y, \lambda)  $$. Therefore

$$
\begin{aligned}
Q(\hat{B}\mid B) &= \sum _{t=1}^T \sum_s P(s_t = i\mid Y, \lambda) \log \hat{b}_{i}(y_t) \\
& = \sum _{t=1}^T \sum_{i=1}^N \gamma_t(i)\log \hat{b}_{i}(y_t)
\end{aligned}
$$

Thus, we summarize the auxiliary function

$$
Q(\hat{\lambda}\mid \lambda)= Q(\hat{\pi}\mid \pi)+ Q(\hat{A}\mid A) + Q(\hat{B}\mid B)
$$


### Maximization step

In the maximization step, we aim to maximize $$Q(\hat{\pi} \mid \pi)$$, $$Q(\hat{A} \mid A)$$ and $$Q(\hat{B} \mid B)$$ with respect to $$\hat{\pi}$$, $$\hat{A}$$ and $$\hat{B}$$ under the following constraints.

$$
\sum\_{i=1}^N \hat{\pi} = 1, \text{ and } \sum_{i=1}^N \hat{A} = 1
$$

Considering the estimation of initial state probabilities $$\mathbf{\hat{\pi}} = {\hat{\pi}_i}$$ , we construct a Lagrange function (or Lagrangian):

$$
Q^*(\hat{\pi} \mid \pi) =\sum_{i=1}^N \gamma_1(i) \log \hat{\pi}_{i} + \eta \left(\sum_{i=1}^N \hat{\pi} - 1 \right)
$$

Differentiating this Lagrangian with respect to individual probability parameter $$\hat{\pi}_i$$  and set it to zero we obtain.

$$
\begin{aligned}
\frac{\partial Q^*(\hat{\pi} \mid \pi)}{\partial \hat{\pi}_i } & = \gamma_1(i) \frac{1}{\hat{\pi}_i} + \eta = 0 \\
\hat{\pi}_i &=  - \frac{1}{\eta}\gamma_1(i)
\end{aligned}
$$

Substituting the above equation into $$\sum_{i=1}^N \hat{\pi} = 1$$ constraint, we obtain:

$$
\begin{aligned}
\sum_{i=1}^N \hat{\pi} &= \sum_{i=1}^N - \frac{1}{\eta}\gamma_1(i) = 1 \\
\Rightarrow \eta &= - \sum_{i=1}^N \gamma_1(i)
\end{aligned}
$$

The ML estimate of new initial state probability is obtained by substituting the above equation into $$ \hat{\pi}_i =  - \frac{1}{\eta}\gamma_1(i)$$:

$$
\hat{\pi}_i = \frac{\gamma_1(i)}{\sum _{i=1}^N \gamma_1(i)} = \gamma_1(i)
$$

In the same manner, we can derive the ML estimates of new state transition probability and new emission probability, which can be shown to be:

$$
\hat{a}_{ij} = \frac{\displaystyle \sum_{t=1}^{T-1}\xi_t(i,j)}{\displaystyle\sum_{t=1}^{T-1}\gamma_t(i)}
$$

And

$$
\hat{b}_i(k) = \frac{\displaystyle\sum_{t=1}^{T}\tau \gamma_t(i)}{\displaystyle\sum_{t=1}^{T} \gamma_t(i)}
$$

where

$$
\tau =
 \begin{cases}
1 \text{ if } y_t = k, \\ 0  \text{ otherwise }
\end{cases}
$$

If we denote the initial model $${\lambda}$$ and the re-estimation model by $$\hat{\lambda}=(\hat{\pi}_i, \hat{a}_{ij},\hat{b}_j(k))$$. Then i t can be shown that either:
 
1. The initial model $$\lambda$$ is a critical point of the likelihood in which case $$\hat{\lambda}= \lambda$$ or
2. $$P(Y \mid \hat\lambda) \leq P(Y \mid \lambda)$$, i.e we have find the better model from which the observation sequence $$Y=y_1,\ldots Y_T$$ is more likely to be produced.
 
Hence we can go on iteractively computing until $$P(Y \mid \hat{\lambda}) $$ is maximazed.

The Baum-Welch Algorithm can be summerized as:

**Require**: $$\lambda \leftarrow \lambda ^{init}$$  

1.  **repeat** 
2.   Compute the forward variable $$\alpha _t(i)$$ from the forward algorithm  
3.   Compute the backward variable $$\beta _t(i)$$ from the backward algorithm 
4.   Compute the occupation probabilities $$\gamma _t(i)$$,  and $$\xi _t(i,j)$$   
5.   Estimate the new HMM parameters $$\hat{\lambda}$$  
6.   Update the HMM parameters   $$\lambda \leftarrow \hat{\lambda}$$
7.  **until** Convergence

### References
1.  L. R. Rabiner, [A tutorial on hidden Markov models and selected applications in speech recognition](http://www.cs.ucsb.edu/~cs281b/papers/HMMs%20-%20Rabiner.pdf), Proceedings of the IEEE, Vol. 77, No. 2, February 1989.
2.  Shinji Watanabe, Jen-Tzung Chien, [Bayesian Speech and Language Processing](https://books.google.co.tz/books/about/Bayesian_Speech_and_Language_Processing.html?id=rEzzCQAAQBAJ&printsec=frontcover&source=kp_read_button&redir_esc=y#v=onepage&q&f=false), Cambridge University Press, 2015.
3. [Viterbi Algorithm in Speech Enhancement and HMM](http://www.vocal.com/echo-cancellation/viterbi-algorithm-in-speech-enhancement-and-hmm/)
4. Nikolai Shokhirev, [Hidden Markov Models](http://www.shokhirev.com/nikolai/abc/alg/hmm/hmm.html)
