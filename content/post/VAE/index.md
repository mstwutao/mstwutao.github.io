---
title: Variational Autoencoder
summary: 
date: 2023-08-01

authors:
  - admin

tags:
  - Generative Models

---

The Variational Autoencoder (VAE) is a probabilistic approach that extends the conventional autoencoder framework by introducing stochasticity in encoding and decoding processes, thus enabling the modeling of complex distributions of data. Originally derived from the principles of Variational Bayesian methods, the VAE framework was associated with autoencoders primarily because certain terms in its objective function could be interpreted analogously to the functions of an encoder and a decoder.

To understand VAE thoroughly, let's begin by deriving its fundamental concepts and then proceed to implement it in Pytorch.

As a type of generative model, VAE aims to capture the underlying distribution of a dataset, which may be intricate and unknown. A commonly used measure of distance between two distributions $p(x)$ and $q(x)$ is the Kullback-Leibler (KL) divergence, which quantifies the difference between them. 
{{< math >}}
$$
\mathcal{D}_{\mathrm{KL}}[p(x) || q(x)] = \mathbb{E}_{p(x)} \left[ \log \frac { p ( x ) } { q ( x ) }  \right]
$$
{{< /math >}}
*Note: the KL divergence is not symmetrical, we will explore their difference in later blogs*

So now, a natural way to model the true data distribution is to minimize the KL divergence between the true data distribution $q(x)$ and the VAE model distribution {{< math >}}$p_{\theta}(x)${{< /math >}}, where {{< math >}}$\theta${{< /math >}} is the parameters of the model which we are trying to optimize.
{{< math >}}
$$
\begin{align}
 \mathcal { D } _ { \mathrm { KL } } [ p_{data} ( x ) || p _ {\theta} ( x ) ] & = \mathbb { E } _ { x \sim p_{data} ( x ) } \left[ \log   p_{data} ( x )  - \log  p _ {\theta} ( x )  \right] \\\\                       & = - H [ p_{data} ( x ) ] - \mathbb { E } _ { x \sim p_{data} ( x ) } \left[ \log p _ {\theta} ( x ) \right] 
\end{align}
$$
{{< /math >}}
Note that $p_{data} ( x )$ is the underlying and unchanging distribution from which our dataset comes, so the entropy of $p _ {data} ( x )$ is a constant, so
{{< math >}}
$$
\min _ { \theta } \mathcal { D } _ { \mathrm { KL } } [ p_{data} ( x ) || p _ {\theta} ( x ) ] = \max _ { \theta } \mathbb { E } _ { p_{data} ( x ) } \left[ \log p _ {\theta} ( x ) \right]
$$
{{< /math >}}
**Minimize the KL divergence of the data distribution and model distribution is equivalent to maximum likelihood method.**

Now, let's see how we model and maximize the likelihood. VAE is a latent variable generative model which learns the distribution of data space $x \in \mathcal { X }$ from a latent space $z \in \mathcal { Z }$, we can define a prior of latent space $p(z)$, which is usually a standard normal distribution, then we can model the data distribution with a complex conditional distribution $p _ { \theta } ( x | z )$, so the model data likelihood can be computed as 

$$p _ { \theta } ( x ) = \int _ { z } p _ { \theta } ( x , z ) \mathrm { d } z = \int _ { z } p _ { \theta } ( x | z ) p ( z ) \mathrm { d } z$$

However, direct maximization of the likelihood is intractable because the intergration. VAE instead optimizes a lower bound of $p _ { \theta } ( x )$, we can derive it using Jenson's Inequality: 


If $f$ is a convex function and $X$ is a random variable, then 

$$
E f ( X ) \geq f ( E X )
$$

the equality holds only when $X = E X$. In our case, 

$$
\begin{align}
 \log p _ {\theta} ( x ) & = \log \int _ { z } p _ { \theta } ( x , z ) \mathrm { d } z  \\\\  & = \log \int _ { z } q _ { \phi } ( z | x )  [ \frac{p _ { \theta } ( x , z )}{q _ { \phi } ( z | x )}] \mathrm { d } z \\\\ & = \log \mathbb { E } _ { q _ { \phi } ( z | x) } [ \frac{p _ { \theta } ( x , z )}{q _ { \phi } ( z | x )}] \\\\ & \geq \mathbb { E } _ { q _ { \phi } ( z | x) } \log [ \frac{p _ { \theta } ( x , z )}{q _ { \phi } ( z | x )}]
\end{align} 
$$

The last line of the derivation is called the **Evidence Lower BOund (ELBO)**, which is used frequently in Variational Inference. It seems confusing what is $q _ { \phi } ( z | x)$? Actually it is an approximate distribution of true posterior $p _ {\theta} ( z | x)$ of latent variable $z$ given datapoint $x$. Let's see where it comes from.

The Variational Autoencoder (VAE) serves not only as a generative model but also as an instance of the Variational Inference family, utilized for inferring data initially. When presented with a raw datapoint $x$, the objective is to learn its representations $z$, encompassing attributes such as shape, size, category, etc. However, directly computing the posterior of latent variables, $p _ { \theta } ( z | x )$, is challenging due to the intractability of $p _ { \theta } ( x )$, as previously discussed. To address this, VAE introduces a recognition model, $q _ { \phi } (z|x)$, which approximates the true posterior, $p _ { \theta } ( z | x )$. The aim is to minimize the Kullback-Leibler (KL) divergence between these distributions.


$$
\begin{align}
 \mathcal { D } _ { \mathrm { KL } } [ q _ { \phi } ( z | x ) || p _ {\theta} ( z | x ) ] & = \mathbb { E } _ { q _ { \phi } ( z | x ) } \left[ \log   q _ { \phi } ( z | x )  - \log  p _ {\theta} ( z | x ) \right] \\\\                       & = \mathbb { E } _ { q _ { \phi } ( z | x ) } \left[ \log   q _ { \phi } ( z | x )  - \log  p _ {\theta} ( x | z ) - \log  p ( z ) \right] + \log p _ { \theta } ( x )  \\\\                                                                                        & = - \mathrm { ELBO } + \log p _ { \theta } ( x ) 
\end{align}
$$


$\log p _ { \theta } (x)$ out of expectation because it does not depend on $z$, rearranging thte equation we obtain

$$
 \mathrm { ELBO } = \log p _ { \theta } (x) - \mathcal { D } _ { \mathrm { KL } } [ q _ { \phi } ( z | x ) || p _ {\theta} ( z | x ) ]
$$

Surprise indeed! It's remarkable that maximizing the Evidence Lower Bound (ELBO) is equivalent to minimizing the Kullback-Leibler (KL) divergence between $q _ { \phi } ( z | x )$ and $p _ {\theta} ( z | x )$, while simultaneously maximizing $\log p _ { \theta } (x)$. This duality in objectives underscores the elegant interplay between approximation and generative modeling within the Variational Autoencoder framework.

So, all the remaining is to maxmize the ELBO, which is tractable under some weak assumptions, let's see how to deal with it.

$$
\begin{align} 
 \mathrm { ELBO } &= \mathbb { E } _ { q _ { \phi } ( z | x) } \log [ \frac{p _ { \theta } ( x , z )}{q _ { \phi } ( z | x )}] \\\\ & = \mathbb { E } _ { q _ { \phi } ( z | x ) } \left[ \log  p _ {\theta} ( x | z ) + \log  p ( z ) - \log q _ { \phi } ( z | x ) \right] \\\\     & = \mathbb { E } _ { q _ { \phi } ( z | x ) } \left[ \log  p _ {\theta} ( x | z ) \right] - \mathcal { D } _ { \mathrm { KL } } [ q _ { \phi } ( z | x ) || p ( z ) ] 
\end{align} 
$$

The first term on RHS is actually the reconstruction error, which is MSE for real value data or cross-entropy for binary value data. The second term is the KL divergence of approximate posterior and prior of latent variables $z$, which can be computed analytically. From the objective function we can see two things:

1. What is  $q _ { \phi } ( z | x )$, given $x$, compute the distribution of $z$; what is $p _ {\theta} ( x | z )$, given $z$, compute the distribution of $x$, if both are implemented by neural network, then they are the encoder and decoder of an Autoencoder, respectively. Now, get the name?

2. Why VAE can generate new data while conventional Autoencoders fail: The first term in the objective is the same as convetional Autoencoder if implemented deterministically, the secret is the second term, VAE forces the mapping from data to latent variables to be as close as a prior, so any time we sample a latent variable from the prior, the decoder knows what to generate, while conventional Autoencoder distribute the latent varibles randomly, there are many gaps between them, if we sample a latent variable from the gap and feed to decoder, the decoder has no idea of it.

Before we implemented VAE, there are still several thing to do. 

Firstly we have a glance of how to compute the $\mathcal { D } _ { \mathrm { KL } } [ q _ { \phi } ( z | x ) || p ( z ) ]$ term. we assume the prior of $z$ is standard Gaussian, $p ( z ) = \mathcal { N } ( 0 , \mathbf { I } )$, this is suitable when implemented VAE by neural networks, because whatever the true prior is, the decoder network can transform the standard Gaussian to it at some layer. So our approximate posterior $q _ { \phi } ( z | x )$ will also take a Guassian distribution form $ \mathcal { N } \left( z ; \boldsymbol { \mu } , \boldsymbol { \sigma } ^ { 2 } \right) $, and the parameters $\boldsymbol { \mu }$ and $\boldsymbol { \sigma }$ is computed by encoder. We compute $\mathcal { D } _ { \mathrm { KL } } [ q _ { \phi } ( z | x ) || p ( z ) ]$ using just simple calculus:

$$
\begin{align} 
 \mathcal { D } _ { \mathrm { KL } } [ q _ { \phi } ( z | x ) || p ( z ) ] & = \mathbb { E } _ { q _ { \phi } ( z | x ) } \left[ \log q _ { \phi } ( z | x ) -  \log  p ( z ) \right] \\\\  & = \int \mathcal { N } \left( z ; \boldsymbol { \mu } , \boldsymbol { \sigma } ^ { 2 } \right) [\log \mathcal { N } \left( z ; \boldsymbol { \mu } , \boldsymbol { \sigma } ^ { 2 } \right) - \log \mathcal { N } ( z ; \mathbf { 0 } , \mathbf { I } )] d \mathbf { z }  \\\\  & = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } \left( - \log \left( \left( \sigma _ { j } \right) ^ { 2 } \right) + \left( \mu _ { j } \right) ^ { 2 } + \left( \sigma _ { j } \right) ^ { 2 } - 1 \right) 
\end{align} 
$$

where $J$ is the dimension of vectors $z$, $\mu _ { j } $ and $\sigma _ { j } $ denote the $j$-th element of mean and variance of $z$, respectively.

We can see the ELBO contains encoder parameters $\phi$ and decoder parameters $\theta$. The gradient with respect to $\theta$ is easy to compute: 

$$ 
\begin{align} 
\nabla _ { \theta } \mathrm {ELBO}  & =  \nabla _ { \theta } \mathbb { E } _ { q _ {\phi} (z | x)}  \log p _ { \theta } ( x | z) \\\\   & =  \mathbb { E } _ { q _ {\phi} (z | x)} [ \nabla _ { \theta }  \log p _ { \theta } ( x | z) ]  \\\\  & \simeq  \frac { 1 } { L } \sum _ { l = 1 } ^ { L } [ \nabla _ { \theta }  \log p _ { \theta } ( x | z ^ { ( l ) }) ]
\end{align}
$$

the last line comes from Monte Carlo estimation, where $ z  ^ { ( l ) } \sim q _ { \phi } \left( z | x \right)$

However, the gradient with respect to $\phi$ need specical handling because common gradient estimator like score function estimator exhibits very high variance thus impractical.

$$ 
\begin{align} 
\nabla _ { \phi } \mathbb { E } _ { q _ { \phi } ( z ) } [ f ( z ) ]  & = \mathbb { E } _ { q _ { \phi } ( z  ) } \left[ f ( z ) \nabla _ { \phi } \log q _ { \phi } ( z ) \right]  \\\\     & \simeq \frac { 1 } { L } \sum _ { l = 1 } ^ { L } [ f ( z ) \nabla _ { \phi } \log q _ { \phi } \left( z ^ { ( l ) } \right) ] 
\end{align}
$$

VAE uses a **reparameterization trick** to derive an unbiased gradient estimator. Instead of sampling $ z \sim q _ { \phi } \left( z | x \right)$ directly, it reparameterize the random variable $\widetilde { z } \sim q _ { \phi } ( z | x )$ using a differentiable transformation $g _ { \phi } (  \epsilon , x )$ with an auxiliary noise variable $\epsilon$.

$$
\widetilde {  z } = g _ { \phi } ( \epsilon , x ) \quad \text { with } \quad \epsilon  \sim p ( \epsilon  )
$$

In the univariate Gaussian case, $z \sim \mathcal { N } \left( \mu , \sigma ^ { 2 } \right)$, we can sample $\epsilon \sim \mathcal { N } ( 0,1 )$ and then use the transformation $z = \mu + \sigma \epsilon$.

In this way, we can compute the gradient with respect to $\phi$ 

$$ 
\begin{align} 
\nabla _ { \phi } \mathrm {ELBO}  & =  \nabla _ { \phi }  \left[ \mathbb { E } _ { q _ { \phi } ( z | x ) } \left[ \log  p _ {\theta} ( x | z ) \right] - \mathcal { D } _ { \mathrm { KL } } [ q _ { \phi } ( z | x ) || p ( z ) ] \right]   \\\\   & =   \nabla _ { \phi }  \left[ \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \left( \log p _ { \boldsymbol { \theta } } \left( x |  z  ^ {  (l)  } \right) \right) - D _ { K L } \left( q _ { \phi } \left( z  | x \right) || p ( z ) \right) \right]
\end{align}
$$

where $ z  ^ {  (l)  } \sim g _ { \phi } \left( x , \epsilon ^ { ( l ) } \right) = \mu  +  \sigma \odot  \epsilon  ^ { ( l ) } \text { where }  \epsilon  ^ { ( l ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) $

*There are several important estimators in machine learning area, we will explore that in later blog.* 

All right, everything is done, the following codes snippets will help to understand the theory discussed above.

First implement encoder network, taking data $x$ as input, output is the mean and standard deviation of $ q _ {\phi} (z | x)$
```python
def encoder(self, x, is_training=True, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        # 1st convolutional layer
        conv1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='same')
        lr_conv1 = tf.nn.leaky_relu(conv1)
        #2nd convolutional layer
        conv2 = tf.layers.conv2d(inputs=lr_conv1, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same')
        lr_conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training = is_training))
        # flatten the convolutional layer
        flat = tf.layers.flatten(lr_conv2)
        #flat = tf.reshape(lr_conv2, [self.batch_size, -1])
        # 1st fully-connected layer
        fc1 = tf.layers.dense(flat, units=1024)
        lr_fc1 = tf.nn.leaky_relu(tf.layers.batch_normalization(fc1, training = is_training))
        # output layer
        out = tf.layers.dense(lr_fc1, units=2*self.z_dim)
        # The mean parameter is unconstrained
        mean = out[:, :self.z_dim]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(out[:, self.z_dim:])
        return mean, stddev

```

Then implement the decoder network, taking latent variables $z$ as input, output the Bernoulli parameter.

```python
def decoder(self, z, is_training=True, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse):
        # 1st fully-connected layer
        fc1 = tf.layers.dense(z, units=1024)
        r_fc1 = tf.nn.relu(tf.layers.batch_normalization(fc1, training=is_training))
        # 2nd fully-connected layer
        fc2 = tf.layers.dense(r_fc1, units=8 * 8 * 128)
        r_fc2 = tf.nn.relu(tf.layers.batch_normalization(fc2, training=is_training))
        # reshape the fully-connected layer
        deflat = tf.reshape(r_fc2, [-1, 8, 8, 128])
        # 1st deconvolutional layer
        deconv1 = tf.layers.conv2d_transpose(inputs=deflat, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='same')
        r_deconv1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=is_training))
        # output layer
        deconv_out = tf.layers.conv2d_transpose(inputs=r_deconv1, filters=1, kernel_size=[4, 4], strides=(2, 2), padding='same')
        out = tf.nn.sigmoid(deconv_out)
        return out

```

We can see how reparameteration trick works and how to implement the objective.

```python
def build_model(self):
    # placeholder
    self.input_x = tf.placeholder(dtype = tf.float32, shape = [None] + self.image_dim, name = 'input_x')
    self.input_z = tf.placeholder(dtype = tf.float32, shape = [None, self.z_dim], name = 'input_z')
    # encoding 
    mu, sigma = self.encoder(self.input_x, is_training = True, reuse = False)
    # reparameterize
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype = tf.float32)
    # decoding 
    outputs = self.decoder(z, is_training = True, reuse = False)
    self.outputs = tf.clip_by_value(outputs, 1e-8, 1 - 1e-8)
    # loss 
    self.nll = -tf.reduce_mean(tf.reduce_sum(self.input_x * tf.log(self.outputs) 
        + (1 - self.input_x) * tf.log(1 - self.outputs), [1, 2]))
    self.KL_div = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) 
        - tf.log(1e-8 + tf.square(sigma)) - 1, [1]))
    ELBO = -(self.nll + self.KL_div)
    self.loss = -ELBO
    # optimizer
    t_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list = t_vars)

```


## Reference

1. [Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).](https://arxiv.org/abs/1312.6114)

2. [Doersch, Carl. "Tutorial on variational autoencoders." arXiv preprint arXiv:1606.05908 (2016).](https://arxiv.org/abs/1606.05908)
