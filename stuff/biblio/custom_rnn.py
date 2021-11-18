from keras import Input
 from keras.layers import Layer, RNN
 from keras.models import Model
 import keras.backend as K

 class CustomLSTMCell(Layer):

     def __init__(self, units, **kwargs):
         self.state_size = [units, units]
         super(CustomLSTMCell, self).__init__(**kwargs)

     def build(self, input_shape):

         self.forget_w = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + input_shape[-1]),
                                         initializer='uniform',
                                         name='forget_w')
         self.forget_b = self.add_weight(shape=(1, self.state_size[0]),
                                         initializer='uniform',
                                         name='forget_b')

         self.input_w1 = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + input_shape[-1]),
                                         initializer='uniform',
                                         name='input_w1')
         self.input_b1 = self.add_weight(shape=(1, self.state_size[0]),
                                         initializer='uniform',
                                         name='input_b1')
         self.input_w2 = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + input_shape[-1]),
                                         initializer='uniform',
                                         name='input_w2')
         self.input_b2 = self.add_weight(shape=(1, self.state_size[0],),
                                         initializer='uniform',
                                         name='input_b2')

         self.output_w = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + input_shape[-1]),
                                         initializer='uniform',
                                         name='output_w')
         self.output_b = self.add_weight(shape=(1, self.state_size[0],),
                                         initializer='uniform',
                                         name='output_b')

         self.built = True

     def merge_with_state(self, inputs):
         self.stateH = K.concatenate([self.stateH, inputs], axis=-1)

     def forget_gate(self):
         forget = K.batch_dot(self.forget_w, self.stateH) + self.forget_b
         forget = K.sigmoid(forget)
         self.stateC = self.stateC * forget

     def input_gate(self):
         candidate = K.batch_dot(self.input_w1, self.stateH) + self.input_b1
         candidate = K.tanh(candidate)

         amount = K.batch_dot(self.input_w2, self.stateH) + self.input_b2
         amount = K.sigmoid(amount)

         self.stateC = self.stateC + amount * candidate

     def output_gate(self):
         self.stateH = K.batch_dot(self.output_w, self.stateH) + self.output_b
         self.stateH = K.sigmoid(self.stateH)

         self.stateH = self.stateH * K.tanh(self.stateC)

     def call(self, inputs, states):

         self.stateH = states[0]
         self.stateC = states[1]

         self.merge_with_state(inputs)
         self.forget_gate()
         self.input_gate()
         self.output_gate()

         return self.stateH, [self.stateH, self.stateC]

 inp = Input(shape=(None, 3))
 lstm = RNN(CustomLSTMCell(10))(inp)

 model = Model(inputs=inp, outputs=lstm)
 inp_value = [[[[1,2,3], [2,3,4], [3,4,5]]]]
 pred = model.predict(inp_value)
 print(pred)
