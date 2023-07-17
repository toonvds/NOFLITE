"""GANITE Codebase.
Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets",
International Conference on Learning Representations (ICLR), 2018.
Paper link: https://openreview.net/forum?id=ByKWUeWA-
Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
ganite.py
Note: GANITE module.
"""

# Necessary packages
import tensorflow as tf
import numpy as np
from benchmarks.ganite.utils_gan import xavier_init, batch_generator

tf.random.set_seed(42)


def ganite(train_x, train_t, train_y, test_x, parameters):
    """GANITE module.

  Args:
    - train_x: features in training data
    - train_t: treatments in training data
    - train_y: observed outcomes in training data
    - test_x: features in testing data
    - parameters: GANITE network parameters
      - h_dim: hidden dimensions
      - batch_size: the number of samples in each batch
      - iterations: the number of iterations for training
      - alpha: hyper-parameter to adjust the loss importance

  Returns:
    - test_y_hat: estimated potential outcome for testing set
  """
    # Parameters
    h_dim = parameters['h_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iterations']
    alpha = parameters['alpha']
    beta = parameters['beta']

    no, dim = train_x.shape

    # Reset graph
    tf.compat.v1.reset_default_graph()

    ## 1. Placeholder
    # 1.1. Feature (X)
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    # 1.2. Treatment (T)
    T = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    # 1.3. Outcome (Y)
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    ## 2. Variables
    # 2.1 Generator
    G_W1 = tf.Variable(xavier_init([(dim + 2), h_dim]))  # Inputs: X + Treatment + Factual outcome
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    # Multi-task outputs for increasing the flexibility of the generator
    G_W31 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b31 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W32 = tf.Variable(xavier_init([h_dim, 1]))
    G_b32 = tf.Variable(tf.zeros(shape=[1]))  # Output: Estimated outcome when t = 0

    G_W41 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b41 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W42 = tf.Variable(xavier_init([h_dim, 1]))
    G_b42 = tf.Variable(tf.zeros(shape=[1]))  # Output: Estimated outcome when t = 1

    # Generator variables
    theta_G = [G_W1, G_W2, G_W31, G_W32, G_W41, G_W42, G_b1, G_b2, G_b31, G_b32, G_b41, G_b42]

    # 2.2 Discriminator
    D_W1 = tf.Variable(
        xavier_init([(dim + 2), h_dim]))  # Inputs: X + Factual outcomes + Estimated counterfactual outcomes
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, 1]))
    D_b3 = tf.Variable(tf.zeros(shape=[1]))

    # Discriminator variables
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # 2.3 Inference network
    I_W1 = tf.Variable(xavier_init([(dim), h_dim]))  # Inputs: X
    I_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    I_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    I_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    # Multi-task outputs for increasing the flexibility of the inference network
    I_W31 = tf.Variable(xavier_init([h_dim, h_dim]))
    I_b31 = tf.Variable(tf.zeros(shape=[h_dim]))

    I_W32 = tf.Variable(xavier_init([h_dim, 1]))
    I_b32 = tf.Variable(tf.zeros(shape=[1]))  # Output: Estimated outcome when t = 0

    I_W41 = tf.Variable(xavier_init([h_dim, h_dim]))
    I_b41 = tf.Variable(tf.zeros(shape=[h_dim]))

    I_W42 = tf.Variable(xavier_init([h_dim, 1]))
    I_b42 = tf.Variable(tf.zeros(shape=[1]))  # Output: Estimated outcome when t = 1

    # Inference network variables
    theta_I = [I_W1, I_W2, I_W31, I_W32, I_W41, I_W42, I_b1, I_b2, I_b31, I_b32, I_b41, I_b42]

    # 3. Definitions of generator, discriminator and inference networks
    # 3.1 Generator
    def generator(x, t, y):
        """Generator function.

    Args:
      - x: features
      - t: treatments
      - y: observed labels

    Returns:
      - G_logit: estimated potential outcomes
    """
        # Concatenate feature, treatments, and observed labels as input
        inputs = tf.concat(axis=1, values=[x, t, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)

        # Estimated outcome if t = 0
        G_h31 = tf.nn.relu(tf.matmul(G_h2, G_W31) + G_b31)
        G_logit1 = tf.matmul(G_h31, G_W32) + G_b32

        # Estimated outcome if t = 1
        G_h41 = tf.nn.relu(tf.matmul(G_h2, G_W41) + G_b41)
        G_logit2 = tf.matmul(G_h41, G_W42) + G_b42

        G_logit = tf.concat(axis=1, values=[G_logit1, G_logit2])
        return G_logit

    # 3.2. Discriminator
    def discriminator(x, t, y, hat_y):
        """Discriminator function.

    Args:
      - x: features
      - t: treatments
      - y: observed labels
      - hat_y: estimated counterfactuals

    Returns:
      - D_logit: estimated potential outcomes
    """
        # Concatenate factual & counterfactual outcomes
        input0 = (1. - t) * y + t * tf.reshape(hat_y[:, 0], [-1, 1])  # if t = 0
        input1 = t * y + (1. - t) * tf.reshape(hat_y[:, 1], [-1, 1])  # if t = 1

        inputs = tf.concat(axis=1, values=[x, input0, input1])

        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        return D_logit

    # 3.3. Inference Nets
    def inference(x):
        """Inference function.

    Args:
      - x: features

    Returns:
      - I_logit: estimated potential outcomes
    """
        I_h1 = tf.nn.relu(tf.matmul(x, I_W1) + I_b1)
        I_h2 = tf.nn.relu(tf.matmul(I_h1, I_W2) + I_b2)

        # Estimated outcome if t = 0
        I_h31 = tf.nn.relu(tf.matmul(I_h2, I_W31) + I_b31)
        I_logit1 = tf.matmul(I_h31, I_W32) + I_b32

        # Estimated outcome if t = 1
        I_h41 = tf.nn.relu(tf.matmul(I_h2, I_W41) + I_b41)
        I_logit2 = tf.matmul(I_h41, I_W42) + I_b42

        I_logit = tf.concat(axis=1, values=[I_logit1, I_logit2])
        return I_logit

    ## Structure
    # 1. Generator
    Y_tilde = generator(X, T, Y)
    # Y_tilde_logit = generator(X, T, Y)
    # Y_tilde = tf.nn.sigmoid(Y_tilde_logit)

    # 2. Discriminator
    D_logit = discriminator(X, T, Y, Y_tilde)

    # 3. Inference network
    Y_hat = inference(X)
    # Y_hat_logit = inference(X)
    # Y_hat = tf.nn.sigmoid(Y_hat_logit)

    ## Loss functions
    # 1. Discriminator loss)
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=T, logits=D_logit))

    # 2. Generator loss
    G_loss_GAN = -D_loss
    G_loss_Factual = tf.reduce_mean(
        tf.nn.l2_loss(Y - (T * tf.reshape(Y_tilde[:, 1], [-1, 1]) + (1. - T) * tf.reshape(Y_tilde[:, 0], [-1, 1]))))
    # G_loss_Factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=Y, logits=(T * tf.reshape(Y_tilde_logit[:, 1], [-1, 1]) + \
    #                       (1. - T) * tf.reshape(Y_tilde_logit[:, 0], [-1, 1]))))

    G_loss = G_loss_Factual + alpha * G_loss_GAN

    # 3. Inference loss
    I_loss1 = tf.reduce_mean(
        tf.nn.l2_loss((T) * Y + (1 - T) * tf.reshape(Y_tilde[:, 1], [-1, 1]) - tf.reshape(Y_hat[:, 1], [-1, 1])))
    I_loss2 = tf.reduce_mean(
        tf.nn.l2_loss((1 - T) * Y + (T) * tf.reshape(Y_tilde[:, 0], [-1, 1]) - tf.reshape(Y_hat[:, 0], [-1, 1])))

    # I_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=(T) * Y + (1 - T) * tf.reshape(Y_tilde[:, 1], [-1, 1]), logits=tf.reshape(Y_hat_logit[:, 1], [-1, 1])))
    # I_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=(1 - T) * Y + (T) * tf.reshape(Y_tilde[:, 0], [-1, 1]), logits=tf.reshape(Y_hat_logit[:, 0], [-1, 1])))

    I_loss = I_loss1 + I_loss2

    ## Solver
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    I_solver = tf.compat.v1.train.AdamOptimizer().minimize(I_loss, var_list=theta_I)

    ## GANITE training
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print('Start training Generator and Discriminator')
    # 1. Train Generator and Discriminator
    for it in range(iterations):

        for _ in range(2):
            # Discriminator training
            X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

        # Generator traininig
        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

        # Check point
        if it % 1000 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) + ', D loss: ' + \
                  str(np.round(D_loss_curr, 4)) + ', G loss: ' + str(np.round(G_loss_curr, 4)))

    print('Start training Inference network')
    # 2. Train Inference network
    for it in range(iterations):

        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)
        _, I_loss_curr = sess.run([I_solver, I_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

        # Check point
        if it % 1000 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) +
                  ', I loss: ' + str(np.round(I_loss_curr, 4)))

    ## Generate the potential outcomes
    test_y_hat = sess.run(Y_hat, feed_dict={X: test_x})

    return test_y_hat


def variables_from_scope(scope_name):
    """
    Returns a list of all trainable variables in a given scope. This is useful when
    you'd like to back-propagate only to weights in one part of the network
    (in our case, the generator or the discriminator).
    """
    return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


def ganite_prob(xtrain, trttrain, ytrain, xtest, parameters):
    """GANITE module.

  Args:
    - train_x: features in training data
    - train_t: treatments in training data
    - train_y: observed outcomes in training data
    - test_x: features in testing data
    - parameters: GANITE network parameters
      - h_dim: hidden dimensions
      - batch_size: the number of samples in each batch
      - iterations: the number of iterations for training
      - alpha: hyper-parameter to adjust the loss importance

  Returns:
    - test_y_hat: estimated potential outcome for testing set
  """
    # Parameters
    h_dim = parameters['h_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iterations']
    alpha = parameters['alpha']
    beta = parameters['beta']
    n_samples = parameters['n_samples']
    input_size = parameters['input_size']

    no, dim = xtrain.shape

    # generator
    # xavier initialization
    def Gcf(x, t, yf, z):
        """combine input feature x,
            treatment t,
            factural outcome yf,
            random generator z,
            and output a random sample of counterfactural outcome
            """
        inputcf = tf.concat([x, t, yf, z], axis=1)
        hidden_layer = tf.compat.v1.layers.dense(inputcf, h_dim,
                                                 kernel_initializer=tf.keras.initializers.glorot_normal(), name="gcf1",
                                                 activation=tf.compat.v1.nn.relu, reuse=None)
        hidden_layer2 = tf.compat.v1.layers.dense(hidden_layer, h_dim,
                                                  kernel_initializer=tf.keras.initializers.glorot_normal(), name="gcf2",
                                                  activation=tf.keras.activations.relu, reuse=None)

        hidden_layer20 = tf.compat.v1.layers.dense(hidden_layer2, h_dim,
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                   name="gcf20", activation=tf.keras.activations.relu, reuse=None)
        hidden_layer21 = tf.compat.v1.layers.dense(hidden_layer2, h_dim,
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                   name="gcf21", activation=tf.keras.activations.relu, reuse=None)

        ycf0 = tf.compat.v1.layers.dense(hidden_layer20, 1, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                         name="ycf0", activation=None, reuse=None)
        ycf1 = tf.compat.v1.layers.dense(hidden_layer21, 1, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                         name="ycf1", activation=None, reuse=None)

        return tf.concat([ycf0, ycf1], 1)

    # descriminator
    def Dcf(x, ycf0, ycf1):
        """combine input feature x,
            counterfactural outcome 0,
            counterfactural outcome 1,
            and output the propability that counterfactural outcome 1 is observed
            """
        inputd = tf.concat([x, ycf0, ycf1], axis=1)
        hidden_layer = tf.compat.v1.layers.dense(inputd, h_dim,
                                                 kernel_initializer=tf.keras.initializers.glorot_normal(), name="d1",
                                                 activation=tf.compat.v1.nn.relu, reuse=None)
        hidden_layer2 = tf.compat.v1.layers.dense(hidden_layer, h_dim,
                                                  kernel_initializer=tf.keras.initializers.glorot_normal(), name="d2",
                                                  activation=tf.compat.v1.nn.relu, reuse=None)
        dlogit = tf.compat.v1.layers.dense(hidden_layer2, 1, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                           name="d3", activation=None, reuse=None)
        return dlogit

    # generator
    def Gite(x, z):
        """combine input feature x,
            random generator z,
            and output a random sample of counterfactural outcome
            """
        inputite = tf.concat([x, z], axis=1)
        hidden_layer = tf.compat.v1.layers.dense(inputite, h_dim,
                                                 kernel_initializer=tf.keras.initializers.glorot_normal(), name="gite1",
                                                 activation=tf.compat.v1.nn.relu, reuse=None)
        hidden_layer2 = tf.compat.v1.layers.dense(hidden_layer, h_dim,
                                                  kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                  name="gite2", activation=tf.keras.activations.relu, reuse=None)

        hidden_layer20 = tf.compat.v1.layers.dense(hidden_layer2, h_dim,
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                   name="gite20", activation=tf.keras.activations.relu, reuse=None)
        hidden_layer21 = tf.compat.v1.layers.dense(hidden_layer2, h_dim,
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                   name="gite21", activation=tf.keras.activations.relu, reuse=None)

        yite0 = tf.compat.v1.layers.dense(hidden_layer20, 1, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                          name="yite0", activation=None, reuse=None)
        yite1 = tf.compat.v1.layers.dense(hidden_layer21, 1, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                          name="yite1", activation=None, reuse=None)

        return tf.concat([yite0, yite1], 1)

    # discriminator
    def Dite(x, ypair):
        """combine input feature x,
            a counterfactural sample y0,y1
            and output the probability this is the true sample
            """
        inputdite = tf.concat([x, ypair], axis=1)
        hidden_layer = tf.compat.v1.layers.dense(inputdite, h_dim,
                                                 kernel_initializer=tf.keras.initializers.glorot_normal(), name="dite1",
                                                 activation=tf.compat.v1.nn.relu, reuse=None)
        hidden_layer2 = tf.compat.v1.layers.dense(hidden_layer, h_dim,
                                                  kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                  name="dite2", activation=tf.keras.activations.relu, reuse=None)
        dite = tf.compat.v1.layers.dense(hidden_layer2, 1, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                         name="dite3", activation=None, reuse=None)
        return dite

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # Placeholders
    yf_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
    t_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
    x_ = tf.compat.v1.placeholder(tf.float32, [None, input_size])

    # generate the cf block
    zcf_ = tf.compat.v1.placeholder(tf.float32, [None, 2])

    # generate the ite block
    zite_ = tf.compat.v1.placeholder(tf.float32, [None, 2])

    # counterfactural generator
    with tf.compat.v1.variable_scope("G") as scope:
        ycf_ = Gcf(x_, t_, yf_, zcf_)

    # all counterfactural outcomes
    ycf0_a = tf.reshape(ycf_[:, 0], [-1, 1])
    ycf1_a = tf.reshape(ycf_[:, 1], [-1, 1])

    # fill in the missing outcome
    ycf0_ = ycf0_a * t_ + yf_ * (1. - t_)
    ycf1_ = ycf1_a * (1. - t_) + yf_ * t_

    # used to train ite block
    ycf_ = tf.concat([ycf0_, ycf1_], axis=1)

    # to construct mse for factural outcomes
    yfpred = ycf0_a * (1. - t_) + ycf1_a * t_

    # descriminator
    with tf.compat.v1.variable_scope("D") as scope:
        dpr_ = Dcf(x_, ycf0_, ycf1_)

    # ite generator
    with tf.compat.v1.variable_scope("gite") as scope:
        ycfpred_ = Gite(x_, zite_)

    # ite discriminator
    with tf.compat.v1.variable_scope("dite") as scope:
        ite_on_fake = Dite(x_, ycfpred_)
        scope.reuse_variables()
        ite_on_real = Dite(x_, ycf_)

    ##conterfactual loss
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(t_, dpr_))
    G_loss_GAN = -D_loss
    MSE_loss = tf.reduce_mean(tf.losses.mean_squared_error(yf_, yfpred))

    G_loss = G_loss_GAN + alpha * MSE_loss

    ##ite loss
    MSE_loss2 = tf.reduce_mean(tf.compat.v1.squared_difference(ycf_, ycfpred_))
    Dite_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(ite_on_fake),
                                                                       ite_on_fake) + tf.nn.sigmoid_cross_entropy_with_logits(
        tf.ones_like(ite_on_real), ite_on_real))
    Gite_loss = -Dite_loss + beta * MSE_loss2

    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer()
    D_step = optimizer.minimize(D_loss, var_list=variables_from_scope("D"))
    G_step = optimizer.minimize(G_loss, var_list=variables_from_scope("G"))
    ite_gstep = optimizer.minimize(Gite_loss, var_list=variables_from_scope("gite"))
    ite_dstep = optimizer.minimize(Dite_loss, var_list=variables_from_scope("dite"))

    # Initializer
    initialize_all = tf.compat.v1.global_variables_initializer()

    siz = len(ytrain)
    sess = tf.compat.v1.Session()
    sess.run(initialize_all)

    dl = []
    gl = []
    gitel = []
    ditel = []

    for t in range(iterations):
        print(end="\r|%-10s|" % ("=" * int(10 * t / (iterations - 1))))
        # randomly generate a minibatch from training set

        # discriminiator
        for i in range(0, 2):
            i = np.random.randint(0, siz, ([batch_size]))
            zcftmp = np.random.uniform(-1.0, 1.0, [batch_size, 2])
            yftmp = ytrain[i, :]
            ttmp = trttrain[i, :]
            xtmp = xtrain[i, :]
            _, dlt = sess.run([D_step, D_loss], feed_dict={yf_: yftmp,
                                                           x_: xtmp,
                                                           zcf_: zcftmp,
                                                           t_: ttmp})
        dl.append(dlt)

        # generator
        i = np.random.randint(0, siz, ([batch_size]))
        zcftmp = np.random.uniform(-1.0, 1.0, [batch_size, 2])
        yftmp = ytrain[i, :]
        ttmp = trttrain[i, :]
        xtmp = xtrain[i, :]

        _, glt = sess.run([G_step, G_loss], feed_dict={yf_: yftmp,
                                                       x_: xtmp,
                                                       zcf_: zcftmp,
                                                       t_: ttmp})
        gl.append(glt)

    for t in range(parameters['iterations']):
        print(end="\r|%-10s|" % ("=" * int(10 * t / (parameters['iterations'] - 1))))
        # randomly generate a minibatch from training set

        # discriminator
        for i in range(0, 2):
            i = np.random.randint(0, siz, ([batch_size]))
            zcftmp = np.random.uniform(-1.0, 1.0, [batch_size, 2])
            zitetmp = np.random.uniform(-1.0, 1.0, [batch_size, 2])
            yftmp = ytrain[i, :]
            ttmp = trttrain[i, :]
            xtmp = xtrain[i, :]

            _, ditelt = sess.run([ite_dstep, Dite_loss], feed_dict={yf_: yftmp,
                                                                    x_: xtmp,
                                                                    zcf_: zcftmp,
                                                                    zite_: zitetmp,
                                                                    t_: ttmp})
        ditel.append(ditelt)

        # generator
        i = np.random.randint(0, siz, ([batch_size]))
        zcftmp = np.random.uniform(-1.0, 1.0, [batch_size, 2])
        zitetmp = np.random.uniform(-1.0, 1.0, [batch_size, 2])
        yftmp = ytrain[i, :]
        ttmp = trttrain[i, :]
        xtmp = xtrain[i, :]

        _, gitelt = sess.run([ite_gstep, Gite_loss], feed_dict={yf_: yftmp,
                                                                x_: xtmp,
                                                                zcf_: zcftmp,
                                                                zite_: zitetmp,
                                                                t_: ttmp})
        gitel.append(gitelt)

    mus_est = np.zeros((len(xtest), n_samples))
    # generate sample from ganite
    for i in range(len(xtest)):
        # sampler for both cf outcomes
        sampsboth = sess.run(ycfpred_, feed_dict={x_: np.tile(xtest[i, :], (n_samples, 1)),
                                                  zite_: np.random.uniform(-1.0, 1.0, [n_samples, 2]),
                                                  })

        samp0 = sampsboth[:, 0].ravel()
        samp1 = sampsboth[:, 1].ravel()

        # estimate the ite
        mus_est[i, :] = samp1 - samp0

    return mus_est
