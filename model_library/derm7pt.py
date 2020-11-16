from model_library import *
from model_library.blocks import *
from tensorflow.keras import *
from tensorflow.keras import layers


class GenericGatedModel(keras.Model):
    """
    Class for a gated model where the same *base model* is used for all the tasks and all the tasks are concatenated
    through gated blocks.
    """
    def __init__(
            self, output_shapes, base_model: type(keras.Model), gates_open=True, input_shape=(224, 224, 3),
            gates_gamma=3.0, name=''
    ):
        super(GenericGatedModel, self).__init__(name=name)
        bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
        self.n_tasks = len(output_shapes)
        self.base_model = base_model
        BN_MOMENTUM = 0.90
        self.bn_momentum = BN_MOMENTUM

        # initialize squeeze nets
        self.task_specific_models = []
        for t, shape in enumerate(output_shapes):
            self.task_specific_models.append(
                base_model(shape, name='t{:02d}_net'.format(t), input_shape=input_shape)
            )
        self.n_blocks = self.task_specific_models[0].n_blocks
        # gates will be added after max pooling
        self.all_gates = []
        self.bn = []
        for b in range(self.n_blocks):
            # add gates for each task
            gate_name = 'gates{}'.format(b + 1)
            gates = []
            conv11 = []
            for t in range(self.n_tasks):
                gates.append(
                    GateLayer(active_task=t, n_tasks=self.n_tasks, gamma=gates_gamma, gates_open=gates_open,
                              name='{}/t{:02d}'.format(gate_name, t))
                )
            # set the gates attribute
            setattr(self, gate_name, gates)
            self.all_gates = [*self.all_gates, *getattr(self, gate_name)]
            # add bn for each block
            if BN_MOMENTUM is not None:
                bn_name = 'bn{}'.format(b + 1)
                bn = layers.BatchNormalization(
                    bn_axis, momentum=BN_MOMENTUM, name=bn_name, scale=False, center=False,
                    renorm=True, renorm_momentum=BN_MOMENTUM
                )
                setattr(self, bn_name, bn)
                self.bn.append(getattr(self, bn_name))

        # setup
        self.weights_alpha = None # self.all_gates.trainable_weights
        self.weights_conv = None # self.task_specific_models.trainable_weights

    def call(self, inputs, training=None, mask=None):
        # set bn to correct value
        for bn in self.bn:
            bn.trainable = training
        for tsp_model in self.task_specific_models:
            for tsp_l in tsp_model.bn:
                if isinstance(tsp_l, layers.BatchNormalization):
                    tsp_l.trainable = training

        x_mid = [inputs] * self.n_tasks
        for b in range(self.n_blocks):
            x_mid = self._call_conv_and_gates(x_mid, b + 1, training=training, debug=False, dropout=False)

        # final block ----------------------
        outputs_final = []
        for t, (current_model, x) in enumerate(zip(self.task_specific_models, x_mid)):
            for layer in current_model.block_final:
                # print(layer.name)  # debug
                x = layer(x, training=training)
            x = current_model.dense(x, training=training)
            outputs_final.append(x)

        return outputs_final

    def _call_conv_and_gates(self, inputs, block_idx, training=False, dropout=False, debug=False):
        outputs_block = []
        for current_model, x in zip(self.task_specific_models, inputs):
            for layer in getattr(current_model, 'block{}'.format(block_idx)):
                x = layer(x, training=training)
                if debug: print(layer.name, x.shape)
            outputs_block.append(x)
        # block gates
        x_concat = tf.concat(outputs_block, axis=-1, name='concat{}'.format(block_idx))
        if self.bn_momentum is not None:
            x_concat = getattr(self, 'bn{}'.format(block_idx))(x_concat, training=training)
        if training and block_idx > 1 and dropout:
            x_concat = tf.nn.dropout(x_concat, 0.15, name='block{}/dropout'.format(block_idx))
        if debug: print('concat{}'.format(block_idx), x_concat.shape)
        outputs_gate = []
        for t, gate in enumerate(getattr(self, 'gates{}'.format(block_idx))):
            x = gate(x_concat, training=training)
            if debug: print(gate.name, x.shape)
            if self.base_model_is_mobilenet:
                conv11 = getattr(self, 'gates{}_conv11'.format(block_idx))[t]
                x = conv11(x)
                if debug: print(conv11.name, x.shape)
            outputs_gate.append(x)
        return outputs_gate

    def get_all_alpha(self, flatten=False, numpy=False):
        all_alpha = []
        for g in self.all_gates:
            tmp = g.get_alpha()
            if flatten:
                tmp = tf.reshape(tmp, [-1])
            else:
                tmp = tf.reshape(tmp, [self.n_tasks, -1])
            if numpy:
                tmp = tmp.numpy()
            all_alpha.append(tmp)
        return all_alpha


class SimpleCNNModel(keras.Model):
    """
    Base class for a simple sequential CNN model. This model is expected to be used as base model for the generic gated model.
    """
    def __init__(self, output_shape, network_setup=None, will_concat=True, input_shape=None, name=''):
        super(SimpleCNNModel, self).__init__(name=name)
        # net setup
        const_filters = 1
        network_setup = ((2, 5, 2), (4, 3, 1), (8, 3, 1), (16, 3, 1))
        self.network_setup = network_setup
        self.n_blocks = len(network_setup)
        # hyper params
        act = 'relu'
        pad = 'valid'
        bias = False
        bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
        # bn_momentum = 0.99
        b = None
        for b, (f, k, s) in enumerate(network_setup):
            k_mp = 2 # 3 if k > 3 else 2
            s_mp = 2
            f_11 = network_setup[b - 1][0] * 4 if b > 0 else None
            # create layers sequence
            if b == 0 or not will_concat:
                block_sequence = [
                    layers.Conv2D(f, k, s, activation=act, padding=pad, use_bias=bias,
                                  name='{}/b{:02d}/conv1'.format(name, b + 1)),
                    layers.MaxPool2D(k_mp, s_mp, padding='valid', name='{}/b{:02d}/pool'.format(name, b + 1))
                ]
            else:
                block_sequence = [
                    layers.Conv2D(f_11, 1, 1, activation=act, padding=pad, use_bias=bias,
                                  name='{}/b{:02d}/conv0'.format(name, b + 1)),
                    layers.Conv2D(f, k, s, activation=act, padding=pad, use_bias=bias,
                                  name='{}/b{:02d}/conv1'.format(name, b + 1)),
                    layers.MaxPool2D(k_mp, s_mp, padding='valid', name='{}/b{:02d}/pool'.format(name, b + 1))
                ]
            # set attribute
            setattr(self, 'block{}'.format(b + 1), block_sequence)

        # block final --------------------------------------------
        b += 1
        f_11 = network_setup[-1][0] * 4
        print('last f_11 {}'.format(f_11))
        self.block_final = [
            layers.Conv2D(f_11, 1, 1, activation=act, padding=pad, use_bias=bias, name='{}/final/conv'.format(name)),
            layers.GlobalAveragePooling2D(name='{}/final/avgpool'.format(name)),
            layers.Dropout(0.3),
            layers.Dense(f_11, activation='relu', ),
            layers.Dropout(0.3)
        ]
        # block final --------------------------------------
        self.dense = layers.Dense(output_shape, name='{}/final/dense'.format(name))
        self.bn = []

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.block1:
            x = layer(x, training=training)
        for layer in self.block2:
            x = layer(x, training=training)
        for layer in self.block3:
            x = layer(x, training=training)
        for layer in self.block_final:
            x = layer(x, training=training)
        x = self.dense(x, training=training)
        return x

    def __repr__(self):
        return self.name