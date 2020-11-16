from model_library import *


class GateLayer(layers.Layer):
    """
    Class defining the Gate Layer.
    Creates a gate layer for the task with index "active_task" in the range of "n_tasks".
    """
    def __init__(self, active_task, n_tasks, gamma=5., gates_open=False, name=None):
        super(GateLayer, self).__init__(name=name)
        # self.units = units
        self.gamma = tf.constant(gamma)
        self.active_task = active_task
        self.n_tasks = n_tasks
        # init alpha values
        self.alpha_raw_before = None
        self.alpha_raw_constant = None
        self.alpha_raw_after = None
        self.gates_open = gates_open

    def build(self, input_shape):
        INIT_CONST = +0.
        INIT_SPREAD = 0.2
        # INITIALIZER = tf.initializers.RandomNormal(mean=INIT_CONST, stddev=INIT_SPREAD, seed=0)
        INITIALIZER = tf.initializers.Constant(value=INIT_CONST)
        concat_shape = input_shape[-1]
        filters_per_task = concat_shape // self.n_tasks
        filters_before = filters_per_task * self.active_task
        filters_after = filters_per_task * (self.n_tasks - self.active_task - 1)

        if filters_before > 0:
            if self.gates_open:
                self.alpha_raw_before = self.add_weight(
                    name='alpha_raw_before',
                    shape=[filters_before],
                    dtype=tf.float32,
                    initializer=INITIALIZER,
                    trainable=True
                )
            else:
                self.alpha_raw_before = self.add_weight(
                    name='alpha_raw_before',
                    shape=[filters_before],
                    dtype=tf.float32,
                    initializer=tf.initializers.constant(-100),
                    trainable=False
                )
        else:
            self.alpha_raw_before = None

        self.alpha_raw_constant = self.add_weight(
            name='alpha_raw_constant',
            shape=[filters_per_task],
            dtype=tf.float32,
            initializer=tf.initializers.constant(100.),
            trainable=False
        )

        if filters_after > 0:
            if self.gates_open:
                self.alpha_raw_after = self.add_weight(
                    name='alpha_raw_after',
                    shape=[filters_after],
                    dtype=tf.float32,
                    initializer=INITIALIZER,
                    trainable=True
                )
            else:
                self.alpha_raw_after = self.add_weight(
                    name='alpha_raw_after',
                    shape=[filters_after],
                    dtype=tf.float32,
                    initializer=tf.initializers.constant(-100),
                    trainable=False
                )
        else:
            self.alpha_raw_after = None

        super(GateLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        alpha = self.get_alpha()
        ans = tf.multiply(alpha, inputs)
        return ans

    def get_alpha(self):
        if self.alpha_raw_after is not None and self.alpha_raw_before is not None:
            all_raw = tf.concat(
                values=[self.alpha_raw_before, self.alpha_raw_constant, self.alpha_raw_after],
                axis=0
            )
        elif self.alpha_raw_before is not None:
            all_raw = tf.concat(
                values=[self.alpha_raw_before, self.alpha_raw_constant],
                axis=0
            )
        elif self.alpha_raw_after is not None:
            all_raw = tf.concat(
                values=[self.alpha_raw_constant, self.alpha_raw_after],
                axis=0
            )
        else:
            raise Exception()

        return gamma_sigmoid(all_raw, self.gamma)

    def __repr__(self):
        return self.name


class GatedBlockGeneric(layers.Layer):
    def __init__(
            self, n_tasks, filters, kernel_size=3, strides=1, n_convolutions=1, init_1_1_conv_filters=None,
            padding='valid', use_bias=True, activation='relu',
            gates_open=True, gates_gamma=3.0, name=''
    ):
        super(GatedBlockGeneric, self).__init__(name=name)
        self.n_tasks = n_tasks
        self.filters = filters
        self.gates_open = gates_open
        self.batch_norm = None
        # conv
        self.task_specific_layers = []
        s11 = filters
        e11 = filters * 2
        e33 = filters * 2
        bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
        for t in range(self.n_tasks):
            self.task_specific_layers.append([
                # layers.BatchNormalization(axis=bn_axis, name= 't{:02d}/bn'.format(t), momentum=BN_MOMENTUM),
                # layers.ReLU(name= name + '/t{:02d}/relu'.format(t))
            ])
            if init_1_1_conv_filters is not None:
               self.task_specific_layers[t].append(
                   layers.Conv2D(init_1_1_conv_filters, kernel_size, padding=padding, use_bias=use_bias,
                                 activation=activation, strides=strides,
                                 name='t{:02d}/init_conv'.format(t))
               )
            for c in range(n_convolutions):
                # if c > 0:
                #     self.task_specific_layers[t].append(
                #         layers.BatchNormalization(axis=bn_axis, name= 't{:02d}/bn{:02d}'.format(t, c),
                #                                   momentum=BN_MOMENTUM, fused=True)
                #     )
                self.task_specific_layers[t].append(
                    layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                  activation=activation,
                                  name= 't{:02d}/conv{:02d}'.format(t, c))
                )
                # self.task_specific_layers[t].append(
                #     FireModule(s11, e11, e33, padding='same', name='t{:02d}/conv{:02d}'.format(t, c))
                # )
            self.task_specific_layers[t].append(
                layers.MaxPool2D(name= '/t{:02d}/pool'.format(t))
            )
            # self.task_specific_layers[t].append(
            #     layers.BatchNormalization(axis=bn_axis, name=name + '/t{:02d}/bn'.format(t)),
            # )
        # concat
        self.concat_all = layers.Concatenate(axis=-1, name=name + '/concat')
        # batch normalization
        # self.batch_norm = layers.BatchNormalization(axis=bn_axis, name=name + '/bn', momentum=BN_MOMENTUM, fused=True)
        # gates
        self.gates = []
        for t in range(self.n_tasks):
            self.gates.append(
                GateLayer(
                    t, self.n_tasks, gamma=gates_gamma, gates_open=gates_open,
                    name=name + '/t{:02d}/gate'.format(t)
                )
            )

    def call(self, inputs, return_all = False, **kwargs):
        x_conv = []
        # convolutions
        for x_in, conv_block in zip(inputs, self.task_specific_layers):
            x_act = x_in
            for conv_layer in conv_block:
                x_act = conv_layer(x_act, **kwargs)
            x_conv.append(x_act)

        # concatenate all
        x_concat = self.concat_all(x_conv, **kwargs)
        # bn
        if self.batch_norm is not None:
            x_concat = self.batch_norm(x_concat, **kwargs)
        # pass through gates
        x_out = []
        for gate in self.gates:
            x_gated = gate(x_concat, **kwargs)
            x_out.append(x_gated)
        # return
        if not return_all:
            return x_out
        else:
            return (x_out, x_conv)