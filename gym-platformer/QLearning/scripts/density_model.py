import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from QLearning.scripts.ex_utils import build_mlp

class Density_Model(object):
    def __init__(self):
        super(Density_Model, self).__init__()

    def receive_tf_sess(self, sess):
        self.sess = sess

    def get_prob(self, state):
        raise NotImplementedError


class Exemplar(Density_Model):
    def __init__(self, ob_dim, hid_dim, learning_rate, kl_weight):
        super(Exemplar, self).__init__()
        self.ob_dim = ob_dim
        self.hid_dim = hid_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight

    def build_computation_graph(self):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            TODO:
                1. self.log_likelihood. shape: (batch_size)
                    - use tf.squeeze
                    - use the discriminator to get the log prob of the discrim_target
                2. self.likelihood. shape: (batch_size)
                    - use tf.squeeze
                    - use the discriminator to get the prob of the discrim_target
                3. self.kl. shape: (batch_size)
                    - simply add the kl divergence between self.encoder1 and 
                        the prior and the kl divergence between self.encoder2 
                        and the prior. Do not average.
                4. self.elbo: 
                    - subtract the kl (weighted by self.kl_weight) from the 
                        log_likelihood, and average over the batch
                5. self.update_op: use the AdamOptimizer with self.learning_rate 
                    to minimize the -self.elbo (Note the negative sign!)

            Hint:
                https://www.tensorflow.org/probability/api_docs/python/tfp/distributions
        """
        self.state1, self.state2 = self.define_placeholders()
        self.encoder1, self.encoder2, self.prior, self.discriminator = self.forward_pass(self.state1, self.state2)
        self.discrim_target = tf.placeholder(shape=[None, 1], name="discrim_target", dtype=tf.float32)

        self.log_likelihood = tf.squeeze(self.discriminator.log_prob(self.discrim_target), axis=1)
        self.likelihood = tf.squeeze(self.discriminator.prob(self.discrim_target), axis=1)
        self.kl = tfp.distributions.kl_divergence(self.encoder1, self.prior) + tfp.distributions.kl_divergence(self.encoder2, self.prior)
        assert len(self.log_likelihood.shape) == len(self.likelihood.shape) == len(self.kl.shape) == 1
    
        self.elbo = tf.reduce_mean(self.log_likelihood - self.kl_weight * self.kl)
        self.update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(-self.elbo)

    def define_placeholders(self):
        state1 = tf.placeholder(shape=[None, self.ob_dim], name="s1", dtype=tf.float32)
        state2 = tf.placeholder(shape=[None, self.ob_dim], name="s2", dtype=tf.float32)
        return state1, state2

    def make_encoder(self, state, z_size, scope, n_layers, hid_size):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state: tf variable
                z_size: output dimension of the encoder network
                scope: scope name
                n_layers: number of layers of the encoder network
                hid_size: hidden dimension of encoder network

            TODO:
                1. z_mean: the output of a neural network that takes the state as input,
                    has output dimension z_size, n_layers layers, and hidden 
                    dimension hid_size
                2. z_logstd: a trainable variable, initialized to 0
                    shape (z_size,)

            Hint: use build_mlp
        """
        z_mean = build_mlp(input_placeholder=state, output_size=z_size, scope=scope, n_layers=n_layers, size=hid_size)
        z_logstd = tf.get_variable('z_logstd', shape=(z_size,), initializer=tf.zeros_initializer())
        return tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.exp(z_logstd))

    def make_prior(self, z_size):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                z_size: output dimension of the encoder network

            TODO:
                prior_mean and prior_logstd are for a standard normal distribution
                    both have dimension z_size
        """
        prior_mean = tf.zeros(z_size)
        prior_logstd = tf.zeros(z_size)
        return tfp.distributions.MultivariateNormalDiag(loc=prior_mean, scale_diag=tf.exp(prior_logstd))

    def make_discriminator(self, z, output_size, scope, n_layers, hid_size):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                z: input to to discriminator network
                output_size: output dimension of discriminator network
                scope: scope name
                n_layers: number of layers of discriminator network
                hid_size: hidden dimension of discriminator network 

            TODO:
                1. logit: the output of a neural network that takes z as input,
                    has output size output_size, n_layers layers, and hidden
                    dimension hid_size

            Hint: use build_mlp
        """
        logit = build_mlp(input_placeholder=z, output_size=output_size, scope=scope, n_layers=n_layers, size=hid_size)
        return tfp.distributions.Bernoulli(logit)

    def forward_pass(self, state1, state2):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: tf variable
                state2: tf variable
            
            encoder1: tfp.distributions.MultivariateNormalDiag distribution
            encoder2: tfp.distributions.MultivariateNormalDiag distribution
            prior: tfp.distributions.MultivariateNormalDiag distribution
            discriminator: tfp.distributions.Bernoulli distribution

            TODO:
                1. z1: sample from encoder1
                2. z2: sample from encoder2
                3. z: concatenate z1 and z2

            Hint: 
                https://www.tensorflow.org/probability/api_docs/python/tfp/distributions
        """
        # Reuse
        make_encoder1 = tf.make_template('encoder1', self.make_encoder)
        make_encoder2 = tf.make_template('encoder2', self.make_encoder)
        make_discriminator = tf.make_template('decoder', self.make_discriminator)

        # Encoder
        encoder1 = make_encoder1(state1, self.hid_dim/2, 'z1', n_layers=2, hid_size=self.hid_dim)
        encoder2 = make_encoder2(state2, self.hid_dim/2, 'z2', n_layers=2, hid_size=self.hid_dim)

        # Prior
        prior = self.make_prior(self.hid_dim/2)

        # Sampled Latent
        z1 = encoder1.sample()
        z2 = encoder2.sample()
        z = tf.concat([z1, z2], axis=1)

        # Discriminator
        discriminator = make_discriminator(z, 1, 'discriminator', n_layers=2, hid_size=self.hid_dim)
        return encoder1, encoder2, prior, discriminator

    def update(self, state1, state2, target):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)
                target: np array (batch_size, 1)

            TODO:
                train the density model and return
                    ll: log_likelihood
                    kl: kl divergence
                    elbo: elbo
        """
        assert state1.ndim == state2.ndim == target.ndim
        assert state1.shape[1] == state2.shape[1] == self.ob_dim
        assert state1.shape[0] == state2.shape[0] == target.shape[0]
        
        _, ll, kl, elbo = self.sess.run([self.update_op, self.log_likelihood, self.kl, self.elbo], feed_dict={self.state1: state1, self.state2: state2, self.discrim_target: target})
        return ll, kl, elbo

    def get_likelihood(self, state1, state2):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)

            TODO:
                likelihood of state1 == state2

            Hint:
                what should be the value of self.discrim_target?
        """
        assert state1.ndim == state2.ndim
        assert state1.shape[1] == state2.shape[1] == self.ob_dim
        assert state1.shape[0] == state2.shape[0]
        target = np.ones((state1.shape[0], 1))
        likelihood = self.sess.run(self.likelihood, feed_dict={self.state1: state1, self.state2: state2, self.discrim_target: target})
        return likelihood

    def get_prob(self, state):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE
        
            args:
                state: np array (batch_size, ob_dim)

            TODO:
                likelihood: 
                    evaluate the discriminator D(x,x) on the same input
                prob:
                    compute the probability density of x from the discriminator
                    likelihood (see homework doc)
        """
        likelihood = self.get_likelihood(state, state)
        # avoid divide by 0 and log(0)
        likelihood = np.clip(np.squeeze(likelihood), 1e-5, 1-1e-5)
        prob = (1 - likelihood) / likelihood
        return prob
