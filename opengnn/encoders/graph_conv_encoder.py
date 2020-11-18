from typing import Union

import tensorflow as tf

from opengnn.encoders.graph_encoder import GraphEncoder
from opengnn.encoders.cnn_encoder import CNNEncoder
from opengnn.utils.cell import build_cell
from opengnn.utils.ops import stack_indices, batch_gather


# initialize with weights so that it start by passing information from the RNN.
def eye_glorot(shape, dtype, partition_info):
    initial_value = tf.glorot_normal_initializer()(shape, dtype, partition_info)
    return initial_value + tf.transpose(tf.eye(shape[-1], shape[0]))


class GraphConvEncoder:
    def __init__(self,
                 base_graph_encoder: GraphEncoder,
                 base_cnn_encoder: CNNEncoder,
                 gnn_input_size: int = None):
        """
        Args:
            base_graph_encoder: A GraphEncoder object that represent that encoder to wrap around.
                All graph propagations will be delegated to this model.
        """

        self.base_graph_encoder = base_graph_encoder
        self.base_cnn_encoder = base_cnn_encoder
        self.gnn_input_size = gnn_input_size
        self.built = False

    def build(self, node_features_size: int, num_edge_types: int):
        self.merge_layer = tf.layers.Dense(
            name = 'merge',
            units=self.gnn_input_size,
            use_bias=False)

        self.base_graph_encoder.build(
            node_features_size, num_edge_types)
        self.base_cnn_encoder.build()

        self.state_map = tf.layers.Dense(
            name="state_map",
            units=self.gnn_input_size,
            use_bias=False,
            kernel_initializer=eye_glorot)

        self.global_filter_gate = tf.layers.Dense(
            name='global_filter_gate',
            units=self.gnn_input_size,
            use_bias=False
        )
        self.global_filter_gate.build((None, self.gnn_input_size * 2))

        self.built = True

    def __call__(self,
                 adj_matrices: tf.SparseTensor,
                 node_features: tf.Tensor,  # Shape: [ batch_size, V, D ]
                 graph_sizes: tf.Tensor,
                 primary_paths: tf.Tensor,
                 primary_path_lengths: tf.Tensor,
                 mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) -> tf.Tensor:
        if not self.built:
            self.build(
                node_features.shape[2].value,
                adj_matrices.get_shape()[1].value)

        # gather representations for the nodes in the pad and do decoding on this path
        primary_path_features = batch_gather(node_features, primary_paths)
        batch_size = tf.shape(node_features, out_type=tf.int64)[0]
        max_num_nodes = tf.shape(node_features, out_type=tf.int64)[1]

        node_representations, graph_state = self.base_graph_encoder(
            adj_matrices=adj_matrices,
            node_features=node_features,
            graph_sizes=graph_sizes,
            mode=mode)

        filter_gate = tf.sigmoid(self.global_filter_gate(tf.concat([
            tf.tile(
                tf.reshape(graph_state, [batch_size, 1, -1]),
                [1, max_num_nodes, 1]), node_representations],
            axis=-1)))
        node_representations = node_representations * filter_gate

        primary_path_representations = batch_gather(node_representations, primary_paths)
        node_features = self.merge_layer(
                tf.concat([primary_path_representations, primary_path_features], axis=-1))
        conv_state, conv_features = self.base_cnn_encoder(
            node_features=node_features,
            seq_lengths=primary_path_lengths,
            mode=mode)

        state = self.state_map(tf.concat([graph_state, conv_state], axis=-1))
        #filter_gate = tf.sigmoid(self.global_filter_gate(tf.concat([
        #    tf.tile(
        #        tf.reshape(graph_state, [batch_size, 1, -1]),
        #        [1, max_num_nodes, 1]), node_representations],
        #    axis=-1)))
        #node_representations = node_representations * filter_gate
        #conv_features = None
        #state = graph_state
        return node_representations, state, conv_features
