import tensorflow as tf


class CNNEncoder:
    def __init__(self,
                 node_feature_size: int,
                 num_kernel_1: int,
                 num_kernel_2: int,
                 kernel_size_1: int,
                 kernel_size_2: int):
        self.node_feature_size = node_feature_size
        self.num_kernel_1 = num_kernel_1
        self.num_kernel_2 = num_kernel_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.built = False

    def build(self) -> None:
        self.w_conv_1 = tf.Variable(tf.truncated_normal(
            [self.kernel_size_1, self.node_feature_size, self.num_kernel_1], stddev=0.1))
        self.w_conv_2 = tf.Variable(tf.truncated_normal(
            [self.kernel_size_2, self.num_kernel_1, self.num_kernel_2], stddev=0.1))
        #self.w_conv_3 = tf.Variable(tf.truncated_normal(
        #    [self.kernel_size_2, self.num_kernel_1, self.num_kernel_2 * 2], stddev=0.1))
        self.kernels = [self.w_conv_1, self.w_conv_2]
        #self.w_conv_attention = tf.Variable(tf.truncated_normal(
        #    [self.kernel_size_2, self.num_kernel_2, 1], stddev=0.1))
        #self.w_conv_3 = tf.Variable(tf.truncated_normal(
        #    [self.kernel_size_2, self.num_kernel_2, self.num_kernel_2], stddev=0.1))
        
        # final linear layer
        self.final_layer = tf.layers.Dense(self.node_feature_size, use_bias=False)
        self.final_layer.build((None, self.node_feature_size))

        #shape = [4000, self.node_feature_size]
        #self.pos_embeddings = tf.get_variable("position_embs", shape=shape, dtype=tf.float32, trainable=True)
        #self.gate_layer = tf.layers.Dense(
        #    1,
        #    activation=tf.nn.sigmoid,
        #    name="cnn_gate_layer")

        #self.output_layer = tf.layers.Dense(
        #    self.node_feature_size,
        #    name="cnn_output_layer")
        #self.built = True

    def __call__(self, node_features: tf.Tensor, seq_lengths: tf.Tensor,
                 mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN):
        if not self.built:
            self.build()
        batch_size = tf.shape(node_features, out_type=tf.int64)[0]
        max_length = tf.shape(node_features, out_type=tf.int64)[1]
        #pos_ids = tf.tile(tf.reshape(tf.range(max_length), (1, -1)), [batch_size, 1])
        #pos_embedding = tf.nn.embedding_lookup(self.pos_embeddings, pos_ids)
        #pos_mask = tf.expand_dims(tf.sequence_mask(seq_lengths, dtype=tf.float32), -1)
        #pos_embedding = pos_embedding * pos_mask

        #node_features = node_features + pos_embedding
        # 和最后一层残差连接
        #input_features = node_features

        next_layer = node_features
        for i in range(2):
            #residual = next_layer
            if mode == tf.estimator.ModeKeys.TRAIN:
                next_layer = tf.layers.dropout(next_layer, rate=0.3, training=True)
            next_layer = tf.nn.conv1d(value=next_layer, filters=self.kernels[i], stride=1, padding='SAME')
            #next_layer = tf.nn.l2_normalize(next_layer, axis=-1)
            #next_layer = gated_linear_units(next_layer)
            #next_layer = (next_layer + residual) * tf.sqrt(0.5)
            #next_layer = tf.nn.l2_normalize(next_layer, axis=-1)
            next_layer = self.parametric_relu(next_layer, i)
            #next_layer = (next_layer + residual) * tf.sqrt(0.5)

        #attention_features = (next_layer + input_features) * tf.sqrt(0.5)
        attention_features = next_layer
        out_features = tf.reduce_mean(attention_features, axis=1)
        out_features = tf.nn.l2_normalize(out_features, axis=1)
        out_features = self.final_layer(out_features)

      

        return out_features, attention_features

    # def get_conv_attention(self, state, input_lengths):
    #     state = tf.transpose(state, perm=[1, 0, 2])
    #     values = tf.nn.l2_normalize(self.attention_features * state, axis=2)
    #     attentions = tf.nn.conv1d(values, filters=self.w_conv_attention, stride=1, padding="SAME")
    #     attentions = tf.nn.softmax(attentions, axis=1)
    #     mask_attentions = attentions * tf.sequence_mask(input_lengths, dtype=tf.int64)
    #     return mask_attentions

    def parametric_relu(self, _x, i):
        alphas = tf.get_variable('alpha_%d'% i, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

def gated_linear_units(inputs):
    input_shape = inputs.get_shape().as_list()
    input_pass = inputs[:, :, 0:int(input_shape[2] / 2)]
    input_gate = inputs[:, :, int(input_shape[2] / 2):]
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)


def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,
                      var_scope_name="conv_layer"):  # padding should take attention

    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * dropout / (kernel_size * in_dim))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs
