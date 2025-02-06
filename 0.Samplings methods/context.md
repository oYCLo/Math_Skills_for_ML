In Machine Learning, Sampling Methods refer to various techniques for selecting data points from a data distribution. These methods play a crucial role in multiple tasks such as data preprocessing, model training, probabilistic inference, and generative models. Depending on the application scenario, the main sampling methods can be categorized as follows:

| **Category** | **Sampling Methods** |
| --- | --- |
| Statistical Sampling | Random Sampling, Uniform Sampling, Stratified Sampling, Systematic Sampling, Importance Sampling |
| Training Data Sampling | Undersampling, Oversampling (SMOTE), Bootstrap Sampling |
| Generative Model Sampling | MCMC (Metropolis-Hastings, Gibbs Sampling), Rejection Sampling, Diffusion Model Sampling, GAN/VAE Sampling |
| Reinforcement Learning Sampling | Experience Replay, ε-Greedy Sampling, Thompson Sampling, Policy Gradient Sampling |
| Approximate Inference Sampling | Importance Sampling, Particle Filtering |

## Monte Carlo Methods Sample
The Monte Carlo Method is a numerical computation method based on random sampling, widely used in probability computation, numerical integration, optimization problems, physical simulation, and machine learning. The core idea of this method is to use a large number of random samples to approximate the mathematical expectation, integral, or optimal solution, especially suitable for high-dimensional problems and problems without analytical solutions.

| **Sampling Method** | **Application Scenario** | **Advantages** | **Disadvantages** |
| --- | --- | --- | --- |
| Direct Sampling | Simple distributions (e.g., uniform, Gaussian) | Low computational cost, easy to implement | Only applicable to specific distributions |
| Rejection Sampling | Arbitrary distributions, but requires auxiliary distribution | Can be used for complex distributions | Inefficient, high rejection rate |
| Importance Sampling | Approximate integration and probability estimation | Suitable for reinforcement learning, Bayesian inference | Depends on suitable \(q(x)\), otherwise high variance |
| MCMC (Metropolis-Hastings, Gibbs) | Complex distributions, such as Bayesian inference | Suitable for high-dimensional spaces | Slow convergence, depends on burn-in |
| Latin Hypercube Sampling | Numerical simulation, experimental design | Ensures multidimensional uniform coverage | Not suitable for discrete distributions |

### Monte Carlo
One of the most common applications of the Monte Carlo method is to compute high-dimensional integrals. For example, to compute the following integral:

$I = \int_{a}^{b} f(x) \,dx$

We can approximate it using random sampling:

$I \approx \frac{b-a}{N} \sum_{i=1}^{N} f(x_i)$

where $x_i$ are random points uniformly sampled in the interval $[a, b]$, and $N$ is the number of samples.

### Rejection Sampling
Rejection sampling is a method used to sample from a complex distribution $p(x)$, especially when direct sampling is not possible but its probability density function (PDF) can be computed. This method indirectly achieves sampling from the target distribution using an easily sampled proposal distribution $q(x)$.

**Input:**

- Target distribution $p(x)$ (needs to be normalized but not easily sampled)
- Proposal distribution $q(x)$ (easily sampled)
- Scaling factor $M$ (such that $p(x) \leq Mq(x)$ for all $x$)

**Steps:**

1. **Choose an appropriate proposal distribution** $q(x)$ and a constant $M$ such that:

$p(x) \leq Mq(x), \, \forall x$

2. **Sample from the proposal distribution**: Draw a sample $x' \sim q(x)$.

3. **Sample from the uniform distribution**: Draw $u \sim U(0,1)$.

4. **Accept or reject the sample**:
    - If $u < \frac{p(x')}{Mq(x')}$, accept $x'$ as a sample from the target distribution $p(x)$.
    - Otherwise, reject $x'$ and return to step 2 to continue sampling.

This process repeats until enough samples are collected.

### Importance Sampling
In probability computation and Monte Carlo methods, Importance Sampling (IS) is a method used to **estimate expectations**, especially when it is difficult to sample directly from the target distribution $p(x)$. The core idea is:
- **Use an easily sampled distribution** $q(x)$ instead of the target distribution $p(x)$ for sampling.
- **Adjust the weight of each sample** so that the final estimate still conforms to the target distribution.

Typically, we want to compute the expectation of a function $f(x)$ under the target distribution $p(x)$:
$E_p[f(x)] = \int f(x) p(x) \, dx$
If it is difficult to sample directly from $p(x)$, we can sample from an easily sampled distribution $q(x)$ and adjust the weights:
$E_p[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i) w_i, \quad w_i = \frac{p(x_i)}{q(x_i)}$
where $w_i$ are called **importance weights**, which are used to adjust the bias of sampling from $q(x)$.

**Advantages and Disadvantages of Importance Sampling**
| **Advantages** | **Disadvantages** |
| -------------- | ----------------- |
| Suitable for situations where it is difficult to sample directly from the target distribution | Requires choosing an appropriate proposal distribution $q(x)$ |
| Widely used in Bayesian inference and reinforcement learning | If $q(x)$ is far from $p(x)$, weights will vary greatly, leading to high variance |
| Suitable for Monte Carlo integration | May perform poorly in high-dimensional spaces |

---

### **Introduction to MCMC (Markov Chain Monte Carlo)**

MCMC (Markov Chain Monte Carlo) is a method used for sampling from complex probability distributions. The core idea is:
- **Construct a Markov chain** whose **stationary distribution** is equal to the target distribution $p(x)$.
- Generate samples through continuous iteration, so that the samples gradually converge to the target distribution.

Two common MCMC algorithms:
1. **Metropolis-Hastings Sampling** (suitable for arbitrary distributions)
2. **Gibbs Sampling** (suitable for situations where conditional distributions are easy to sample)

---

**Metropolis-Hastings Sampling**
1. **Choose an easily sampled proposal distribution** $q(x' | x)$ to generate a new sample $x'$.
2. **Calculate the acceptance probability**:
   $A = \min\left( 1, \frac{p(x') q(x | x')}{p(x) q(x' | x)} \right)$
3. **Accept or reject the new sample**:
   - Accept $x'$ as the new sample with probability $A$.
   - Otherwise, keep the current $x$ unchanged and continue sampling.

---

## **4. Gibbs Sampling**
### **Basic Idea**
Gibbs Sampling is a special MCMC method suitable for **high-dimensional joint distributions**, but **each variable's conditional distribution is easy to sample**.

Assume there are two variables $x$ and $y$, with a joint distribution $p(x, y)$, but it is difficult to sample directly:
1. **Alternately sample** each variable's conditional distribution:
   $x^{(t+1)} \sim p(x | y^{(t)})$
   $y^{(t+1)} \sim p(y | x^{(t+1)})$
2. **Repeat the process until the samples converge to the joint distribution**.

| **Algorithm** | **Applicable Situation** | **Advantages and Disadvantages** |
| ------------- | ----------------------- | ------------------------------- |
| **Metropolis-Hastings (MH)** | Suitable for **any probability distribution** | Easy to implement, but may converge slowly in high-dimensional cases |
| **Gibbs Sampling** | **Conditional distributions are easy to sample** | Fast convergence, but requires conditional distributions to be analyzable |

---