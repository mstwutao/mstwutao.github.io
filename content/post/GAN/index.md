---
title: Generative Adversarial Networks
summary: 
date: 2023-07-01

authors:
  - admin

tags:
  - Generative Models

---

Generative Adversarial Networks(GAN) has become one of the most powerful techniques in machine learning since its emergence in 2014. The interesting idea of training methods and the flexible design of the objective function make GAN have numerous variants.

Let's first interpret GAN from perspectives and then see how to implement GAN in Pytorch:

## Game Theory

GAN consists of two models: a generative model $G$ that tries to generate realistic samples, and a discriminative model $D$ that tries to classify data from both $G$ and dataset. The training procedure for $G$ is to maximize the probability of $D$ making a mistake, and $D$ is to maximize the probability of correctly classify all samples it receives. This framework corresponds to a minimax two-player game, the generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles.

Let's formulate GAN in mathematics now. Both $G : Z \rightarrow X$ and $D : X \rightarrow [0, 1]$ are represented by neural networks, like other common generative models, $G$ takes vectors of noise $z \sim p(z)$ as input, the output $G(z)$ are the fake samples with the same size as data from dataset, $D$ recieves samples from both dataset and $G$, and it outputs the probability of the samples it receives are whether real or fake.

Hence, the objective of $D$ is:

$$
\begin{align}
V(D) = \max  \mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } ) } [ \log D ( \boldsymbol { x } ) ] + \mathbb { E } _ { \boldsymbol { z } \sim p ( \boldsymbol { z } ) } [ \log ( 1 - D ( G ( \boldsymbol { z } ) ) ) ]
\end{align}
$$

The objective of $G$ is:

$$
\begin{align}
V(G) & = \max \mathbb { E } _ { \boldsymbol { z } \sim p ( \boldsymbol { z } ) } [ \log (D ( G ( \boldsymbol { z } ) ) ) ] \\\\ & = \min \mathbb { E } _ { \boldsymbol { z } \sim p ( \boldsymbol { z } ) } [ \log ( 1 - D ( G ( \boldsymbol { z } ) ) ) ] \\\\ & = \min \mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } ) } [ \log D ( \boldsymbol { x } ) ] + \mathbb { E } _ { \boldsymbol { z } \sim p ( \boldsymbol { z } ) } [ \log ( 1 - D ( G ( \boldsymbol { z } ) ) ) ]
\end{align}
$$

Equation (4) holds because $\mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } ) } [ \log D ( \boldsymbol { x } ) ]$ does not depend on $G$, so we can write the objective of GAN as:

$$
\begin{align}
\min _ { G } \max _ { D } V ( D , G ) =  \mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } ) } [ \log D ( \boldsymbol { x } ) ] + \mathbb { E } _ { \boldsymbol { z } \sim p ( \boldsymbol { z } ) } [ \log ( 1 - D ( G ( \boldsymbol { z } ) ) ) ]
\end{align}
$$


## Divergence Minimization

As a generative model, $G$ tries to capture the true data distribution, hence it seeks to minimize a kind of divergence between the model distribution $p _ { g }$ and data distribution $p _ { \mathrm { data } }$. But how can $G$ achieve it? 

The training process of GAN is to train $G$ and $D$ in turn, when training $G$ we keep $D$ fixed and vice versa, before the training converges, what is the optimal $D$ for a fixed $G$?

$$
\begin{align}
\max _ { D } V ( D , G ) & = \max \mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } ) } [ \log D ( \boldsymbol { x } ) ] + \mathbb { E } _ { \boldsymbol { z } \sim p ( \boldsymbol { z } ) } [ \log ( 1 - D ( G ( \boldsymbol { z } ) ) ) ] \\\\ & = \max \int _ { \boldsymbol { x } } p _ { \text { data } } ( \boldsymbol { x } ) \log ( D ( \boldsymbol { x } ) ) d x + \int _ { \boldsymbol { z } } p  ( \boldsymbol { z } ) \log ( 1 - D ( g ( \boldsymbol { z } ) ) ) d z \\\\ & = \max \int _ { \boldsymbol { x } } p _ { \text { data } } ( \boldsymbol { x } ) \log ( D ( \boldsymbol { x } ) ) + p _ { g } ( \boldsymbol { x } ) \log ( 1 - D ( \boldsymbol { x } ) ) d x
\end{align}
$$

Using simple calculus, we know function $ a \log ( y ) + b \log ( 1 - y )$ achieves its maximum in $[0, 1]$ at $\frac { a } { a + b }$, hence the optimal $D$ is 

$$
\begin{align}
D  ^ { * } ( \boldsymbol { x } ) = \frac { p _ { \text {data} } ( \boldsymbol { x } ) } { p _ { \text {data} } ( \boldsymbol { x } ) + p _ { g } ( \boldsymbol { x } ) }
\end{align}
$$

If at every step, we train $D$ to the optimum, then what will the objective of $G$ like?

$$
\begin{align}
\min _ { G } V ( D , G ) & = \min \mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } ) } \left[ \log D ^ { * } ( \boldsymbol { x } ) \right] + \mathbb { E } _ { \boldsymbol { z } \sim p (z)} \left[ \log \left( 1 - D  ^ { * } ( G ( \boldsymbol { z } ) ) \right) \right] \\\\ & = \min \mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } ) } \left[ \log D ^ { * } ( \boldsymbol { x } ) \right] + \mathbb { E } _ { \boldsymbol { x } \sim p _ { g } ( \boldsymbol { x } )} \left[ \log \left( 1 - D ^ { * } ( \boldsymbol { x } ) \right) \right] \\\\ & = \min \mathbb { E } _ { \boldsymbol { x } \sim p _ { \text { data } } ( \boldsymbol { x } )} \left[ \log \frac { p _ { \text { data } } ( \boldsymbol { x } ) } { P _ { \text { data } } ( \boldsymbol { x } ) + p _ { g } ( \boldsymbol { x } ) } \right] \\\\ & + \mathbb { E } _ { \boldsymbol { x } \sim p _ { g } ( \boldsymbol { x } )} \left[ \log \frac { p _ { g } ( \boldsymbol { x } ) } { p _ { \text { data } } ( \boldsymbol { x } ) + p _ { g } ( \boldsymbol { x } ) } \right] \\\\ & = - \log ( 4 ) + \mathcal { D } _ { \mathrm { KL } } \left( p _ { \text { data } } || \frac { p _ { \text { data } } + p _ { g } } { 2 } \right) + \mathcal { D } _ { \mathrm { KL } } \left( p _ { g } || \frac { p _ { \text { data } } + p _ { g } } { 2 } \right) \\\\ & = - \log ( 4 ) + 2 \cdot \mathcal { D } _ { \mathrm { JSD } } \left( p _ { \text { data } } || p _ { g } \right)
\end{align}
$$

Now we can see that $G$ is minimizing the Jensen– Shannon divergence between the model’s distribution and the true data distribution. All right, let's see the Tensorflow implementation of GAN to better understand the theory discussed above. The implementation is based on Deep Convolutional GAN (DCGAN) because the original GAN's training is very instable.

## Tensorflow implementation

The generator part, it takes a vector of $z$ as input, output an fake image sample.

```python
def generator(self, z, is_training=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        z = tf.reshape(z, [-1, 1, 1, self.config.z_dim])
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(z, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=is_training), 0.2)
    
        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=is_training), 0.2)
    
        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=is_training), 0.2)
    
        # output layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, self.image_dim[-1], [4, 4], strides=(2, 2), padding='same')
        out = tf.nn.sigmoid(conv4)

        return out
```

The discriminator receive an image, outputs the probability that image is real.
```python
def discriminator(self, x, is_training=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = tf.nn.leaky_relu(conv1, 0.2)
    
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=is_training), 0.2)
    
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=is_training), 0.2)
    
        # output layer
        logits = tf.layers.conv2d(lrelu3, 1, [4, 4], strides=(1, 1), padding='valid')
        out = tf.nn.sigmoid(logits)
    
        return out, logits
```
The loss for both networks are pretty self-explainatory.
```python
def build_model(self):
    # placeholder
    self.input_x = tf.placeholder(dtype=tf.float32, shape=[None] + self.image_dim, name='input_x')
    self.input_z = tf.placeholder(dtype=tf.float32, shape=[None, self.config.z_dim], name='input_z')

    # generate fake data
    self.g_fake = self.generator(self.input_z) 

    # discrinminate real and fake data
    self.d_real, self.d_real_logits = self.discriminator(self.input_x)
    self.d_fake, self.d_fake_logits = self.discriminator(self.g_fake, reuse=True)

    # testing generated images
    self.generated_images = self.generator(self.input_z, is_training=False, reuse=True)

    # loss
    with tf.variable_scope("loss"):
        # loss for each network
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logits, labels=tf.ones_like(self.d_real_logits)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.zeros_like(self.d_fake_logits)))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.ones_like(self.d_fake_logits)))
    
        # optimizer
    with tf.variable_scope("optimizer"):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=0.5).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=0.5).minimize(self.g_loss, var_list=g_vars)
```


## Reference

1. [Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

2. [Jolicoeur-Martineau, Alexia. "GANs beyond divergence minimization." arXiv preprint arXiv:1809.02145 (2018).](https://arxiv.org/pdf/1809.02145.pdf)
