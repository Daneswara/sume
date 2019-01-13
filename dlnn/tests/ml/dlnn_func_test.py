import keras
import numpy
from keras import Input
from keras import backend as K
from keras.engine.saving import load_model
from keras.initializers import Initializer
from keras.layers import Lambda, Reshape

from dlnn.Dlnn import Dlnn
from dlnn.layer.Conv2D import AvgFilter, MaxFilter, StdDevFilter, Conv2D
from dlnn.layer.MergeCategorical import MergeCategorical
from dlnn.layer.Scaling import Scaling
from dlnn.layer.Tiling import Tiling
from dlnn.tests.ml.activation_test import layer_2_activation, layer_4_activation, layer_7_activation, \
    layer_11_a_activation, layer_11_b_activation, layer_11_c_activation
from dlnn.tests.ml.conv_test import layer_1_conv, layer_3_conv, layer_6_conv
from dlnn.tests.ml.elm_func_test import layer_9_flatten, layer_10_a_dense, layer_12_a_dense, layer_10_b_dense, \
    layer_12_b_dense, layer_10_c_dense, layer_12_c_dense, layer_13_concatenate, layer_14_reshape, \
    layer_15_merge_categorical
from dlnn.tests.ml.pooling_test import layer_5_pool, layer_8_pool
from dlnn.tests.ml.repos_helper import corpus_data, label_init, corpus_label, normalized, corr_step_1, corr_step_2, \
    corr_step_3, corr_step_4, corr_step_5, corr_step_6, corr_step_7, corr_step_8, corr_step_8_full, corr_step_9, \
    corr_step_10_a_bias_init, corr_step_10_a_kernel_init, corr_step_10_b_kernel_init, corr_step_10_b_bias_init, \
    corr_step_10_c_bias_init, corr_step_10_c_kernel_init, corr_step_10_a, corr_step_10_b, corr_step_10_c, \
    corr_step_11_a, corr_step_11_b, corr_step_11_c
from dlnn.tests.ml.testcase import TestCase
from dlnn.util import to_categorical
from dlnn.util.Initializers import Unifinv

inputs = Input(shape=(corpus_data.shape[-1],))
scale = Lambda(lambda x: x * 1.0 / 300.0)(inputs)
reshape = Reshape([1, 1, 6])(scale)
tile = Lambda(lambda x: K.tile(x, (1, 1, 6, 1)))(reshape)
step_1_conv = layer_1_conv()(tile)
step_2_activation = layer_2_activation()(step_1_conv)
step_3_conv = layer_3_conv()(step_2_activation)
step_4_activation = layer_4_activation()(step_3_conv)
step_5_pool = layer_5_pool(2)(step_4_activation)
step_6_conv = layer_6_conv()(step_5_pool)
step_7_activation = layer_7_activation()(step_6_conv)
step_8_pool = layer_8_pool(1)(step_7_activation)
step_9_flatten = layer_9_flatten()(step_8_pool)
step_10_a_dense = layer_10_a_dense()(step_9_flatten)
step_11_a_activation = layer_11_a_activation()(step_10_a_dense)
step_12_a_dense = layer_12_a_dense()(step_11_a_activation)
step_10_b_dense = layer_10_b_dense()(step_9_flatten)
step_11_b_activation = layer_11_b_activation()(step_10_b_dense)
step_12_b_dense = layer_12_b_dense()(step_11_b_activation)
step_10_c_dense = layer_10_c_dense()(step_9_flatten)
step_11_c_activation = layer_11_c_activation()(step_10_c_dense)
step_12_c_dense = layer_12_c_dense()(step_11_c_activation)
step_13_concatenate = layer_13_concatenate()([step_12_a_dense, step_12_b_dense, step_12_c_dense])
step_14_reshape = layer_14_reshape()(step_13_concatenate)
step_15_output = layer_15_merge_categorical()(step_14_reshape)


class CustomInitializer(Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __init__(self, val) -> None:
        self.val = val
        super(Initializer, self).__init__()

    def __call__(self, shape, dtype=None):
        return K.variable(self.val)

    def get_config(self):
        config = {
            'val': self.val,
        }
        base_config = super(CustomInitializer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'val' in config:
            config['val'] = numpy.array(config['val']['value'])

        return super().from_config(config)


class DlnnDump(Dlnn):

    def __init__(self, network_name='network-dump', scale_min1=0.0, scale_max1=0.0, scale_min2=0.0, scale_max2=0.0,
                 conv_1_window=0, conv_2_window=0, conv_3_window=0, pool_1_size=0, pool_2_size=0, elm_1_dense_1_units=0,
                 elm_2_dense_1_units=0, elm_3_dense_1_units=0, elm_1_dense_1_kernel_min=0.0,
                 elm_1_dense_1_kernel_max=0.0, elm_2_dense_1_kernel_min=0.0, elm_2_dense_1_kernel_max=0.0,
                 elm_3_dense_1_kernel_min=0.0, elm_3_dense_1_kernel_max=0.0, elm_1_dense_1_bias_min=0.0,
                 elm_1_dense_1_bias_max=0.0, elm_2_dense_1_bias_min=0.0, elm_2_dense_1_bias_max=0.0,
                 elm_3_dense_1_bias_min=0.0, elm_3_dense_1_bias_max=0.0):
        super(DlnnDump, self).__init__(network_name, scale_min1, scale_max1, scale_min2, scale_max2, conv_1_window,
                                       conv_2_window,
                                       conv_3_window, pool_1_size, pool_2_size, elm_1_dense_1_units,
                                       elm_2_dense_1_units,
                                       elm_3_dense_1_units, elm_1_dense_1_kernel_min, elm_1_dense_1_kernel_max,
                                       elm_2_dense_1_kernel_min, elm_2_dense_1_kernel_max, elm_3_dense_1_kernel_min,
                                       elm_3_dense_1_kernel_max, elm_1_dense_1_bias_min, elm_1_dense_1_bias_max,
                                       elm_2_dense_1_bias_min, elm_2_dense_1_bias_max, elm_3_dense_1_bias_min,
                                       elm_3_dense_1_bias_max)

    def build_model(self):
        from keras import Input
        from keras.activations import sigmoid
        from keras.initializers import Zeros
        from keras.layers import Reshape, Activation, MaxPooling2D, Flatten, Dense, Concatenate

        self.layer['input'] = Input(
            shape=(self.input_shape,))

        self.layer['pre_scaling'] = Scaling(
            scale_min1=self.scale_min1,
            scale_max1=self.scale_max1,
            scale_min2=self.scale_min2,
            scale_max2=self.scale_max2,
            name='pre_scaling')(self.layer['input'])

        self.layer['pre_reshaping'] = Reshape(
            target_shape=[1, 1, self.input_shape],
            name='pre_reshaping')(self.layer['pre_scaling'])

        self.layer['pre_tiling'] = Tiling(
            units=self.input_shape,
            name='pre_tiling')(self.layer['pre_reshaping'])

        self.layer['cnn_conv_1'] = Conv2D(
            filters=[AvgFilter(), MaxFilter(), StdDevFilter()],
            window=self.conv_1_window,
            padding='same',
            use_bias=False,
            kernel_size=(self.input_shape, self.input_shape),
            data_format='channels_first',
            name='cnn_conv_1')(self.layer['pre_tiling'])

        self.layer['cnn_activation_1'] = Activation(
            activation=sigmoid,
            name='cnn_activation_1')(self.layer['cnn_conv_1'])

        self.layer['cnn_conv_2'] = Conv2D(
            filters=[AvgFilter(), MaxFilter(), StdDevFilter()],
            window=self.conv_2_window,
            padding='same',
            use_bias=False,
            kernel_size=(self.input_shape, self.input_shape),
            data_format='channels_first',
            name='cnn_conv_2')(self.layer['cnn_activation_1'])

        self.layer['cnn_activation_2'] = Activation(
            activation=sigmoid,
            name='cnn_activation_2')(self.layer['cnn_conv_2'])

        self.layer['cnn_pooling_1'] = MaxPooling2D(
            pool_size=(self.pool_1_size, self.pool_1_size),
            padding='same',
            data_format='channels_first',
            name='cnn_pooling_1')(self.layer['cnn_activation_2'])

        self.layer['cnn_conv_3'] = Conv2D(
            filters=[AvgFilter(), MaxFilter(), StdDevFilter()],
            window=self.conv_3_window,
            padding='same',
            use_bias=False,
            kernel_size=(self.input_shape, self.input_shape),
            data_format='channels_first',
            name='cnn_conv_3')(self.layer['cnn_pooling_1'])

        self.layer['cnn_activation_3'] = Activation(
            activation=sigmoid,
            name='cnn_activation_3')(self.layer['cnn_conv_3'])

        self.layer['cnn_pooling_2'] = MaxPooling2D(
            pool_size=(self.pool_2_size, self.pool_2_size),
            padding='same',
            data_format='channels_first',
            name='cnn_pooling_2')(self.layer['cnn_activation_3'])

        self.layer['bridge_flatten'] = Flatten(
            name='bridge_flatten')(self.layer['cnn_pooling_2'])

        self.layer['elm_1_dense_1'] = Dense(
            units=self.elm_1_dense_1_units,
            activation=None,
            use_bias=True,
            kernel_initializer=CustomInitializer(corr_step_10_a_kernel_init),
            bias_initializer=CustomInitializer(corr_step_10_a_bias_init),
            trainable=False,
            name='elm_1_dense_1')(self.layer['bridge_flatten'])

        self.layer['elm_1_activation_1'] = Activation(
            activation=sigmoid,
            name='elm_1_activation_1')(self.layer['elm_1_dense_1'])

        self.layer['elm_1_dense_2'] = Dense(
            units=self.category_num,
            activation=None,
            use_bias=False,
            kernel_initializer=Zeros(),
            trainable=False,
            name='elm_1_dense_2')(self.layer['elm_1_activation_1'])

        self.layer['elm_2_dense_1'] = Dense(
            units=self.elm_2_dense_1_units,
            activation=None,
            use_bias=True,
            kernel_initializer=CustomInitializer(corr_step_10_b_kernel_init),
            bias_initializer=CustomInitializer(corr_step_10_b_bias_init),
            trainable=False,
            name='elm_2_dense_1')(self.layer['bridge_flatten'])

        self.layer['elm_2_activation_1'] = Activation(
            activation=sigmoid,
            name='elm_2_activation_1')(self.layer['elm_2_dense_1'])

        self.layer['elm_2_dense_2'] = Dense(
            units=self.category_num,
            activation=None,
            use_bias=False,
            kernel_initializer=Zeros(),
            trainable=False,
            name='elm_2_dense_2')(self.layer['elm_2_activation_1'])

        self.layer['elm_3_dense_1'] = Dense(
            units=self.elm_3_dense_1_units,
            activation=None,
            use_bias=True,
            kernel_initializer=CustomInitializer(corr_step_10_c_kernel_init),
            bias_initializer=CustomInitializer(corr_step_10_c_bias_init),
            trainable=False,
            name='elm_3_dense_1')(self.layer['bridge_flatten'])

        self.layer['elm_3_activation_1'] = Activation(
            activation=sigmoid,
            name='elm_3_activation_1')(self.layer['elm_3_dense_1'])

        self.layer['elm_3_dense_2'] = Dense(
            units=self.category_num,
            activation=None,
            use_bias=False,
            kernel_initializer=Zeros(),
            trainable=False,
            name='elm_3_dense_2')(self.layer['elm_3_activation_1'])

        self.layer['fully_connected_concat'] = Concatenate(
            name='fully_connected_concat')(
            [self.layer['elm_1_dense_2'], self.layer['elm_2_dense_2'], self.layer['elm_3_dense_2']])

        self.layer['fully_connected_reshape'] = Reshape(
            target_shape=(self.fully_connected_num, self.category_num),
            name='fully_connected_reshape')(self.layer['fully_connected_concat'])

        self.layer['fully_connected_merge'] = MergeCategorical(
            categorical_length=self.category_num,
            name='fully_connected_merge')(self.layer['fully_connected_reshape'])

    def get_model(self):
        return load_model(self.network_path, custom_objects={'Scaling': Scaling,
                                                             'Tiling': Tiling,
                                                             'AvgFilter': AvgFilter,
                                                             'MaxFilter': MaxFilter,
                                                             'StdDevFilter': StdDevFilter,
                                                             'Conv2D': Conv2D,
                                                             'MergeCategorical': MergeCategorical,
                                                             'Unifinv': Unifinv,
                                                             'CustomInitializer': CustomInitializer})


class DlnnFunctionalTest(TestCase):
    def test_functional_dlnn(self):
        from keras import Model
        from dlnn.util import MoorePenrose
        #
        # Feed Beta
        #
        feed = Model(inputs=inputs, outputs=step_11_a_activation)
        output = feed.predict(corpus_data)
        w_10_a = feed.get_layer(index=13).get_weights()
        w_12_a = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        feed = Model(inputs=inputs, outputs=step_11_b_activation)
        output = feed.predict(corpus_data)
        w_10_b = feed.get_layer(index=13).get_weights()
        w_12_b = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        feed = Model(inputs=inputs, outputs=step_11_c_activation)
        output = feed.predict(corpus_data)
        w_10_c = feed.get_layer(index=13).get_weights()
        w_12_c = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]

        #
        # Training Model
        #
        network = Model(inputs=inputs, outputs=step_15_output)
        network.compile(optimizer=keras.optimizers.RMSprop(lr=0.0, rho=0.0, epsilon=None, decay=0.0),
                        loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_accuracy, keras.metrics.mape])
        for i in range(25):
            layer = network.get_layer(index=i).get_weights()
            self.assertIsNotNone(layer)
            # print("step_%d" % (i + 1), layer)
        network.get_layer(index=13).set_weights(w_10_a)
        network.get_layer(index=14).set_weights(w_10_b)
        network.get_layer(index=15).set_weights(w_10_c)
        network.get_layer(index=19).set_weights(w_12_a)
        network.get_layer(index=20).set_weights(w_12_b)
        network.get_layer(index=21).set_weights(w_12_c)
        network.fit(corpus_data, to_categorical(label_init, 3), batch_size=corpus_data.shape[0])
        self.assertTrue(numpy.allclose(w_10_a[0], network.get_layer(index=13).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_10_b[0], network.get_layer(index=14).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_10_c[0], network.get_layer(index=15).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_12_a[0], network.get_layer(index=19).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_12_b[0], network.get_layer(index=20).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_12_c[0], network.get_layer(index=21).get_weights()[0], rtol=0))
        result = network.predict(corpus_data, batch_size=corpus_data.shape[0])
        # print(result.argmax(axis=1))
        self.assertIsNotNone(result)

    def test_baked_dlnn(self):
        self.assertTrue(True)
        from dlnn.Dlnn import Dlnn
        from dlnn.Dlnn import DLNN_DEFAULT_CONFIG
        train_eval = Dlnn(**DLNN_DEFAULT_CONFIG).train(corpus_data, corpus_label - 1)
        self.assertIsNotNone(train_eval)
        # print(train_eval)

    def test_baked_dlnn_value(self):
        self.assertTrue(True)
        from dlnn.Dlnn import Dlnn
        from dlnn.Dlnn import DLNN_DEFAULT_CONFIG
        yc = keras.utils.to_categorical(label_init, len(numpy.unique(label_init)))
        dlnn = Dlnn(**DLNN_DEFAULT_CONFIG)
        network = dlnn.get_model()
        train_eval = network.evaluate(corpus_data, yc)
        self.assertIsNotNone(train_eval)
        # network.summary()
        # print(train_eval)

        from keras import Model
        layer_name = 'pre_tiling'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, normalized, rtol=1e-6))

        layer_name = 'cnn_conv_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_1, rtol=1e-6))

        layer_name = 'cnn_activation_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_2, rtol=1e-6))

        layer_name = 'cnn_conv_2'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_3, rtol=1e-6))

        layer_name = 'cnn_activation_2'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_4, rtol=1e-6))

        layer_name = 'cnn_pooling_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_5, rtol=1e-6))

        layer_name = 'cnn_conv_3'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_6, rtol=1e-6))

        layer_name = 'cnn_activation_3'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_7, rtol=1e-6))

        layer_name = 'cnn_pooling_2'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_8, rtol=1e-6))
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_8_full, rtol=1e-6))

        layer_name = 'bridge_flatten'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_9, rtol=1e-6))

    def test_baked_dlnndump(self):
        from keras import Model
        from dlnn.util import MoorePenrose
        #
        # Feed Beta
        #
        feed = Model(inputs=inputs, outputs=step_11_a_activation)
        output = feed.predict(corpus_data)
        w_10_a = feed.get_layer(index=13).get_weights()
        w_12_a = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        feed = Model(inputs=inputs, outputs=step_11_b_activation)
        output = feed.predict(corpus_data)
        w_10_b = feed.get_layer(index=13).get_weights()
        w_12_b = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        feed = Model(inputs=inputs, outputs=step_11_c_activation)
        output = feed.predict(corpus_data)
        w_10_c = feed.get_layer(index=13).get_weights()
        w_12_c = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]

        self.assertTrue(True)
        from dlnn.Dlnn import DLNN_DEFAULT_CONFIG
        yc = keras.utils.to_categorical(label_init, len(numpy.unique(label_init)))
        dlnn = DlnnDump(**DLNN_DEFAULT_CONFIG)
        # dlnn.train(corpus_data, corpus_label - 1)
        network = dlnn.get_model()
        train_eval = network.evaluate(corpus_data, yc)
        self.assertIsNotNone(train_eval)
        # network.summary()
        # print(train_eval)

        from keras import Model
        layer_name = 'pre_tiling'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, normalized, rtol=1e-6))

        layer_name = 'cnn_conv_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_1, rtol=1e-6))

        layer_name = 'cnn_activation_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_2, rtol=1e-6))

        layer_name = 'cnn_conv_2'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_3, rtol=1e-6))

        layer_name = 'cnn_activation_2'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_4, rtol=1e-6))

        layer_name = 'cnn_pooling_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_5, rtol=1e-6))

        layer_name = 'cnn_conv_3'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_6, rtol=1e-6))

        layer_name = 'cnn_activation_3'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_7, rtol=1e-6))

        layer_name = 'cnn_pooling_2'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output[0], corr_step_8, rtol=1e-6))
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_8_full, rtol=1e-6))

        layer_name = 'bridge_flatten'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_9, rtol=1e-6))

        layer_name = 'elm_1_dense_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_10_a, rtol=1e-6))
        self.assertTrue(
            numpy.allclose(corr_step_10_a_kernel_init, network.get_layer(layer_name).get_weights()[0], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(corr_step_10_a_bias_init, network.get_layer(layer_name).get_weights()[1], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(w_10_a[0], network.get_layer(layer_name).get_weights()[0], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(w_10_a[1], network.get_layer(layer_name).get_weights()[1], rtol=1e-6))

        layer_name = 'elm_1_activation_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_11_a, rtol=1e-6))

        layer_name = 'elm_2_dense_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_10_b, rtol=1e-6))
        self.assertTrue(
            numpy.allclose(corr_step_10_b_kernel_init, network.get_layer(layer_name).get_weights()[0], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(corr_step_10_b_bias_init, network.get_layer(layer_name).get_weights()[1], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(w_10_b[0], network.get_layer(layer_name).get_weights()[0], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(w_10_b[1], network.get_layer(layer_name).get_weights()[1], rtol=1e-6))

        layer_name = 'elm_2_activation_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_11_b, rtol=1e-6))

        layer_name = 'elm_3_dense_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_10_c, rtol=1e-6))
        self.assertTrue(
            numpy.allclose(corr_step_10_c_kernel_init, network.get_layer(layer_name).get_weights()[0], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(corr_step_10_c_bias_init, network.get_layer(layer_name).get_weights()[1], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(w_10_c[0], network.get_layer(layer_name).get_weights()[0], rtol=1e-6))
        self.assertTrue(
            numpy.allclose(w_10_c[1], network.get_layer(layer_name).get_weights()[1], rtol=1e-6))

        layer_name = 'elm_3_activation_1'
        intermediate = Model(inputs=network.input,
                             outputs=network.get_layer(layer_name).output)
        intermediate_output = intermediate.predict(corpus_data)
        self.assertTrue(numpy.allclose(intermediate_output, corr_step_11_c, rtol=1e-6))

        layer_name = 'elm_1_dense_2'
        self.assertTrue(
            numpy.allclose(w_12_a[0], network.get_layer(layer_name).get_weights()[0], rtol=1e-6))

        layer_name = 'elm_2_dense_2'
        self.assertTrue(
            numpy.allclose(w_12_b[0], network.get_layer(layer_name).get_weights()[0], rtol=1e-6))

        layer_name = 'elm_3_dense_2'
        self.assertTrue(
            numpy.allclose(w_12_c[0], network.get_layer(layer_name).get_weights()[0], rtol=1e-6))

    def test_testing(self):
        self.assertTrue(True)
        from dlnn.Dlnn import Dlnn
        from dlnn.Dlnn import DLNN_DEFAULT_CONFIG
        dlnn = Dlnn(**DLNN_DEFAULT_CONFIG)
        network = dlnn.get_model()
        result = network.predict(corpus_data[0].reshape((1, corpus_data.shape[1])), batch_size=corpus_data.shape[0])
        print(result.argmax(axis=1))
