class RNN(base_layer.Layer):
  """
  stolen from github
  """


  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               **kwargs):
    if isinstance(cell, (list, tuple)):
      cell = StackedRNNCells(cell)
    if 'call' not in dir(cell):
      raise ValueError('Argument `cell` should have a `call` method. '
                       f'The RNN was passed: cell={cell}')
    if 'state_size' not in dir(cell):
      raise ValueError('The RNN cell should have a `state_size` attribute '
                       '(tuple of integers, one integer per RNN state). '
                       f'Received: cell={cell}')
    # If True, the output for masked timestep will be zeros, whereas in the
    # False case, output from previous timestep is returned for masked timestep.
    self.zero_output_for_mask = kwargs.pop('zero_output_for_mask', False)


    if 'input_shape' not in kwargs and (
        'input_dim' in kwargs or 'input_length' in kwargs):
      input_shape = (kwargs.pop('input_length', None),
                     kwargs.pop('input_dim', None))
      kwargs['input_shape'] = input_shape


    super(RNN, self).__init__(**kwargs)
    self.cell = cell
    self.return_sequences = return_sequences
    self.return_state = return_state
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.unroll = unroll
    self.time_major = time_major


    self.supports_masking = True
    # The input shape is unknown yet, it could have nested tensor inputs, and
    # the input spec will be the list of specs for nested inputs, the structure
    # of the input_spec will be the same as the input.
    self.input_spec = None
    self.state_spec = None
    self._states = None
    self.constants_spec = None
    self._num_constants = 0


    if stateful:
      if tf.distribute.has_strategy():
        raise ValueError('Stateful RNNs (created with `stateful=True`) '
                         'are not yet supported with tf.distribute.Strategy.')


  @property
  def _use_input_spec_as_call_signature(self):
    if self.unroll:
      # When the RNN layer is unrolled, the time step shape cannot be unknown.
      # The input spec does not define the time step (because this layer can be
      # called with any time step value, as long as it is not None), so it
      # cannot be used as the call function signature when saving to SavedModel.
      return False
    return super(RNN, self)._use_input_spec_as_call_signature


  @property
  def states(self):
    if self._states is None:
      state = tf.nest.map_structure(lambda _: None, self.cell.state_size)
      return state if tf.nest.is_nested(self.cell.state_size) else [state]
    return self._states


  @states.setter
  # Automatic tracking catches "self._states" which adds an extra weight and
  # breaks HDF5 checkpoints.
  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def states(self, states):
    self._states = states


  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    # Check whether the input shape contains any nested shapes. It could be
    # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
    # inputs.
    try:
      input_shape = tf.TensorShape(input_shape)
    except (ValueError, TypeError):
      # A nested tensor input
      input_shape = tf.nest.flatten(input_shape)[0]


    batch = input_shape[0]
    time_step = input_shape[1]
    if self.time_major:
      batch, time_step = time_step, batch


    if _is_multiple_state(self.cell.state_size):
      state_size = self.cell.state_size
    else:
      state_size = [self.cell.state_size]


    def _get_output_shape(flat_output_size):
      output_dim = tf.TensorShape(flat_output_size).as_list()
      if self.return_sequences:
        if self.time_major:
          output_shape = tf.TensorShape(
              [time_step, batch] + output_dim)
        else:
          output_shape = tf.TensorShape(
              [batch, time_step] + output_dim)
      else:
        output_shape = tf.TensorShape([batch] + output_dim)
      return output_shape


    if getattr(self.cell, 'output_size', None) is not None:
      # cell.output_size could be nested structure.
      output_shape = tf.nest.flatten(tf.nest.map_structure(
          _get_output_shape, self.cell.output_size))
      output_shape = output_shape[0] if len(output_shape) == 1 else output_shape
    else:
      # Note that state_size[0] could be a tensor_shape or int.
      output_shape = _get_output_shape(state_size[0])


    if self.return_state:
      def _get_state_shape(flat_state):
        state_shape = [batch] + tf.TensorShape(flat_state).as_list()
        return tf.TensorShape(state_shape)
      state_shape = tf.nest.map_structure(_get_state_shape, state_size)
      return generic_utils.to_list(output_shape) + tf.nest.flatten(state_shape)
    else:
      return output_shape


  def compute_mask(self, inputs, mask):
    # Time step masks must be the same for each input.
    # This is because the mask for an RNN is of size [batch, time_steps, 1],
    # and specifies which time steps should be skipped, and a time step
    # must be skipped for all inputs.
    # TODO(scottzhu): Should we accept multiple different masks?
    mask = tf.nest.flatten(mask)[0]
    output_mask = mask if self.return_sequences else None
    if self.return_state:
      state_mask = [None for _ in self.states]
      return [output_mask] + state_mask
    else:
      return output_mask


  def build(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
      # The input_shape here could be a nest structure.


    # do the tensor_shape to shapes here. The input could be single tensor, or a
    # nested structure of tensors.
    def get_input_spec(shape):
      """Convert input shape to InputSpec."""
      if isinstance(shape, tf.TensorShape):
        input_spec_shape = shape.as_list()
      else:
        input_spec_shape = list(shape)
      batch_index, time_step_index = (1, 0) if self.time_major else (0, 1)
      if not self.stateful:
        input_spec_shape[batch_index] = None
      input_spec_shape[time_step_index] = None
      return InputSpec(shape=tuple(input_spec_shape))


    def get_step_input_shape(shape):
      if isinstance(shape, tf.TensorShape):
        shape = tuple(shape.as_list())
      # remove the timestep from the input_shape
      return shape[1:] if self.time_major else (shape[0],) + shape[2:]


    def get_state_spec(shape):
      state_spec_shape = tf.TensorShape(shape).as_list()
      # append batch dim
      state_spec_shape = [None] + state_spec_shape
      return InputSpec(shape=tuple(state_spec_shape))


    # Check whether the input shape contains any nested shapes. It could be
    # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
    # inputs.
    try:
      input_shape = tf.TensorShape(input_shape)
    except (ValueError, TypeError):
      # A nested tensor input
      pass


    if not tf.nest.is_nested(input_shape):
      # This indicates the there is only one input.
      if self.input_spec is not None:
        self.input_spec[0] = get_input_spec(input_shape)
      else:
        self.input_spec = [get_input_spec(input_shape)]
      step_input_shape = get_step_input_shape(input_shape)
    else:
      if self.input_spec is not None:
        self.input_spec[0] = tf.nest.map_structure(get_input_spec, input_shape)
      else:
        self.input_spec = generic_utils.to_list(
            tf.nest.map_structure(get_input_spec, input_shape))
      step_input_shape = tf.nest.map_structure(get_step_input_shape, input_shape)


    # allow cell (if layer) to build before we set or validate state_spec.
    if isinstance(self.cell, base_layer.Layer) and not self.cell.built:
      with backend.name_scope(self.cell.name):
        self.cell.build(step_input_shape)
        self.cell.built = True


    # set or validate state_spec
    if _is_multiple_state(self.cell.state_size):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]


    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      self._validate_state_spec(state_size, self.state_spec)
    else:
      if tf.nest.is_nested(state_size):
        self.state_spec = tf.nest.map_structure(get_state_spec, state_size)
      else:
        self.state_spec = [
            InputSpec(shape=[None] + tf.TensorShape(dim).as_list())
            for dim in state_size
        ]
      # ensure the generated state_spec is correct.
      self._validate_state_spec(state_size, self.state_spec)
    if self.stateful:
      self.reset_states()
    self.built = True


  @staticmethod
  def _validate_state_spec(cell_state_sizes, init_state_specs):
    """Validate the state spec between the initial_state and the state_size.

    Args:
      cell_state_sizes: list, the `state_size` attribute from the cell.
      init_state_specs: list, the `state_spec` from the initial_state that is
        passed in `call()`.

    Raises:
      ValueError: When initial state spec is not compatible with the state size.
    """
    validation_error = ValueError(
        'An `initial_state` was passed that is not compatible with '
        '`cell.state_size`. Received `state_spec`={}; '
        'however `cell.state_size` is '
        '{}'.format(init_state_specs, cell_state_sizes))
    flat_cell_state_sizes = tf.nest.flatten(cell_state_sizes)
    flat_state_specs = tf.nest.flatten(init_state_specs)


    if len(flat_cell_state_sizes) != len(flat_state_specs):
      raise validation_error
    for cell_state_spec, cell_state_size in zip(flat_state_specs,
                                                flat_cell_state_sizes):
      if not tf.TensorShape(
          # Ignore the first axis for init_state which is for batch
          cell_state_spec.shape[1:]).is_compatible_with(
              tf.TensorShape(cell_state_size)):
        raise validation_error


  @doc_controls.do_not_doc_inheritable
  def get_initial_state(self, inputs):
    get_initial_state_fn = getattr(self.cell, 'get_initial_state', None)


    if tf.nest.is_nested(inputs):
      # The input are nested sequences. Use the first element in the seq to get
      # batch size and dtype.
      inputs = tf.nest.flatten(inputs)[0]


    input_shape = tf.shape(inputs)
    batch_size = input_shape[1] if self.time_major else input_shape[0]
    dtype = inputs.dtype
    if get_initial_state_fn:
      init_state = get_initial_state_fn(
          inputs=None, batch_size=batch_size, dtype=dtype)
    else:
      init_state = _generate_zero_filled_state(batch_size, self.cell.state_size,
                                               dtype)
    # Keras RNN expect the states in a list, even if it's a single state tensor.
    if not tf.nest.is_nested(init_state):
      init_state = [init_state]
    # Force the state to be a list in case it is a namedtuple eg LSTMStateTuple.
    return list(init_state)


  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    inputs, initial_state, constants = _standardize_args(inputs,
                                                         initial_state,
                                                         constants,
                                                         self._num_constants)


    if initial_state is None and constants is None:
      return super(RNN, self).__call__(inputs, **kwargs)


    # If any of `initial_state` or `constants` are specified and are Keras
    # tensors, then add them to the inputs and temporarily modify the
    # input_spec to include them.


    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      additional_inputs += initial_state
      self.state_spec = tf.nest.map_structure(
          lambda s: InputSpec(shape=backend.int_shape(s)), initial_state)
      additional_specs += self.state_spec
    if constants is not None:
      additional_inputs += constants
      self.constants_spec = [
          InputSpec(shape=backend.int_shape(constant)) for constant in constants
      ]
      self._num_constants = len(constants)
      additional_specs += self.constants_spec
    # additional_inputs can be empty if initial_state or constants are provided
    # but empty (e.g. the cell is stateless).
    flat_additional_inputs = tf.nest.flatten(additional_inputs)
    is_keras_tensor = backend.is_keras_tensor(
        flat_additional_inputs[0]) if flat_additional_inputs else True
    for tensor in flat_additional_inputs:
      if backend.is_keras_tensor(tensor) != is_keras_tensor:
        raise ValueError(
            'The initial state or constants of an RNN layer cannot be '
            'specified via a mix of Keras tensors and non-Keras tensors '
            '(a "Keras tensor" is a tensor that was returned by a Keras layer '
            ' or by `Input` during Functional model construction). '
            f'Received: initial_state={initial_state}, constants={constants}')


    if is_keras_tensor:
      # Compute the full input spec, including state and constants
      full_input = [inputs] + additional_inputs
      if self.built:
        # Keep the input_spec since it has been populated in build() method.
        full_input_spec = self.input_spec + additional_specs
      else:
        # The original input_spec is None since there could be a nested tensor
        # input. Update the input_spec to match the inputs.
        full_input_spec = generic_utils.to_list(
            tf.nest.map_structure(lambda _: None, inputs)) + additional_specs
      # Perform the call with temporarily replaced input_spec
      self.input_spec = full_input_spec
      output = super(RNN, self).__call__(full_input, **kwargs)
      # Remove the additional_specs from input spec and keep the rest. It is
      # important to keep since the input spec was populated by build(), and
      # will be reused in the stateful=True.
      self.input_spec = self.input_spec[:-len(additional_specs)]
      return output
    else:
      if initial_state is not None:
        kwargs['initial_state'] = initial_state
      if constants is not None:
        kwargs['constants'] = constants
      return super(RNN, self).__call__(inputs, **kwargs)


  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)


    inputs, initial_state, constants = self._process_inputs(
        inputs, initial_state, constants)


    self._maybe_reset_cell_dropout_mask(self.cell)
    if isinstance(self.cell, StackedRNNCells):
      for cell in self.cell.cells:
        self._maybe_reset_cell_dropout_mask(cell)


    if mask is not None:
      # Time step masks must be the same for each input.
      # TODO(scottzhu): Should we accept multiple different masks?
      mask = tf.nest.flatten(mask)[0]


    if tf.nest.is_nested(inputs):
      # In the case of nested input, use the first element for shape check.
      input_shape = backend.int_shape(tf.nest.flatten(inputs)[0])
    else:
      input_shape = backend.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]
    if self.unroll and timesteps is None:
      raise ValueError('Cannot unroll a RNN if the '
                       'time dimension is undefined. \n'
                       '- If using a Sequential model, '
                       'specify the time dimension by passing '
                       'an `input_shape` or `batch_input_shape` '
                       'argument to your first layer. If your '
                       'first layer is an Embedding, you can '
                       'also use the `input_length` argument.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a `shape` '
                       'or `batch_shape` argument to your Input layer.')


    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training


    # TF RNN cells expect single tensor as state instead of list wrapped tensor.
    is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
    # Use the __call__ function for callable objects, eg layers, so that it
    # will have the proper name scopes for the ops, etc.
    cell_call_fn = self.cell.__call__ if callable(self.cell) else self.cell.call
    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError(
            f'RNN cell {self.cell} does not support constants. '
            f'Received: constants={constants}')


      def step(inputs, states):
        constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
        states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type


        states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
        output, new_states = cell_call_fn(
            inputs, states, constants=constants, **kwargs)
        if not tf.nest.is_nested(new_states):
          new_states = [new_states]
        return output, new_states
    else:


      def step(inputs, states):
        states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
        output, new_states = cell_call_fn(inputs, states, **kwargs)
        if not tf.nest.is_nested(new_states):
          new_states = [new_states]
        return output, new_states
    last_output, outputs, states = backend.rnn(
        step,
        inputs,
        initial_state,
        constants=constants,
        go_backwards=self.go_backwards,
        mask=mask,
        unroll=self.unroll,
        input_length=row_lengths if row_lengths is not None else timesteps,
        time_major=self.time_major,
        zero_output_for_mask=self.zero_output_for_mask)


    if self.stateful:
      updates = [
          tf.compat.v1.assign(self_state, tf.cast(state, self_state.dtype))
          for self_state, state in zip(
              tf.nest.flatten(self.states), tf.nest.flatten(states))
      ]
      self.add_update(updates)


    if self.return_sequences:
      output = backend.maybe_convert_to_ragged(
          is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards)
    else:
      output = last_output


    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return generic_utils.to_list(output) + states
    else:
      return output


  def _process_inputs(self, inputs, initial_state, constants):
    # input shape: `(samples, time (padded with zeros), input_dim)`
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    if (isinstance(inputs, collections.abc.Sequence)
        and not isinstance(inputs, tuple)):
      # get initial_state from full input spec
      # as they could be copied to multiple GPU.
      if not self._num_constants:
        initial_state = inputs[1:]
      else:
        initial_state = inputs[1:-self._num_constants]
        constants = inputs[-self._num_constants:]
      if len(initial_state) == 0:
        initial_state = None
      inputs = inputs[0]


    if self.stateful:
      if initial_state is not None:
        # When layer is stateful and initial_state is provided, check if the
        # recorded state is same as the default value (zeros). Use the recorded
        # state if it is not same as the default.
        non_zero_count = tf.add_n([tf.math.count_nonzero(s)
                                   for s in tf.nest.flatten(self.states)])
        # Set strict = True to keep the original structure of the state.
        initial_state = tf.compat.v1.cond(non_zero_count > 0,
                                          true_fn=lambda: self.states,
                                          false_fn=lambda: initial_state,
                                          strict=True)
      else:
        initial_state = self.states
      initial_state = tf.nest.map_structure(
          # When the layer has a inferred dtype, use the dtype from the cell.
          lambda v: tf.cast(v, self.compute_dtype or self.cell.compute_dtype),
          initial_state
      )
    elif initial_state is None:
      initial_state = self.get_initial_state(inputs)


    if len(initial_state) != len(self.states):
      raise ValueError(f'Layer has {len(self.states)} '
                       f'states but was passed {len(initial_state)} initial '
                       f'states. Received: initial_state={initial_state}')
    return inputs, initial_state, constants


  def _validate_args_if_ragged(self, is_ragged_input, mask):
    if not is_ragged_input:
      return


    if mask is not None:
      raise ValueError(f'The mask that was passed in was {mask}, which '
                       'cannot be applied to RaggedTensor inputs. Please '
                       'make sure that there is no mask injected by upstream '
                       'layers.')
    if self.unroll:
      raise ValueError('The input received contains RaggedTensors and does '
                       'not support unrolling. Disable unrolling by passing '
                       '`unroll=False` in the RNN Layer constructor.')


  def _maybe_reset_cell_dropout_mask(self, cell):
    if isinstance(cell, DropoutRNNCellMixin):
      cell.reset_dropout_mask()
      cell.reset_recurrent_dropout_mask()


  def reset_states(self, states=None):
    """Reset the recorded states for the stateful RNN layer.

    Can only be used when RNN layer is constructed with `stateful` = `True`.
    Args:
      states: Numpy arrays that contains the value for the initial state, which
        will be feed to cell at the first time step. When the value is None,
        zero filled numpy array will be created based on the cell state size.

    Raises:
      AttributeError: When the RNN layer is not stateful.
      ValueError: When the batch size of the RNN layer is unknown.
      ValueError: When the input numpy array is not compatible with the RNN
        layer state, either size wise or dtype wise.
    """
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    spec_shape = None
    if self.input_spec is not None:
      spec_shape = tf.nest.flatten(self.input_spec[0])[0].shape
    if spec_shape is None:
      # It is possible to have spec shape to be None, eg when construct a RNN
      # with a custom cell, or standard RNN layers (LSTM/GRU) which we only know
      # it has 3 dim input, but not its full shape spec before build().
      batch_size = None
    else:
      batch_size = spec_shape[1] if self.time_major else spec_shape[0]
    if not batch_size:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the batch size by passing a '
                       '`batch_shape` argument to your Input layer.')
    # initialize state if None
    if tf.nest.flatten(self.states)[0] is None:
      if getattr(self.cell, 'get_initial_state', None):
        flat_init_state_values = tf.nest.flatten(self.cell.get_initial_state(
            inputs=None, batch_size=batch_size,
            # Use variable_dtype instead of compute_dtype, since the state is
            # stored in a variable
            dtype=self.variable_dtype or backend.floatx()))
      else:
        flat_init_state_values = tf.nest.flatten(_generate_zero_filled_state(
            batch_size, self.cell.state_size,
            self.variable_dtype or backend.floatx()))
      flat_states_variables = tf.nest.map_structure(
          backend.variable, flat_init_state_values)
      self.states = tf.nest.pack_sequence_as(self.cell.state_size,
                                             flat_states_variables)
      if not tf.nest.is_nested(self.states):
        self.states = [self.states]
    elif states is None:
      for state, size in zip(tf.nest.flatten(self.states),
                             tf.nest.flatten(self.cell.state_size)):
        backend.set_value(
            state,
            np.zeros([batch_size] + tf.TensorShape(size).as_list()))
    else:
      flat_states = tf.nest.flatten(self.states)
      flat_input_states = tf.nest.flatten(states)
      if len(flat_input_states) != len(flat_states):
        raise ValueError(f'Layer {self.name} expects {len(flat_states)} '
                         f'states, but it received {len(flat_input_states)} '
                         f'state values. States received: {states}')
      set_value_tuples = []
      for i, (value, state) in enumerate(zip(flat_input_states,
                                             flat_states)):
        if value.shape != state.shape:
          raise ValueError(
              f'State {i} is incompatible with layer {self.name}: '
              f'expected shape={(batch_size, state)} '
              f'but found shape={value.shape}')
        set_value_tuples.append((state, value))
      backend.batch_set_value(set_value_tuples)


  def get_config(self):
    config = {
        'return_sequences': self.return_sequences,
        'return_state': self.return_state,
        'go_backwards': self.go_backwards,
        'stateful': self.stateful,
        'unroll': self.unroll,
        'time_major': self.time_major
    }
    if self._num_constants:
      config['num_constants'] = self._num_constants
    if self.zero_output_for_mask:
      config['zero_output_for_mask'] = self.zero_output_for_mask


    config['cell'] = generic_utils.serialize_keras_object(self.cell)
    base_config = super(RNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


  @classmethod
  def from_config(cls, config, custom_objects=None):
    from keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cell = deserialize_layer(config.pop('cell'), custom_objects=custom_objects)
    num_constants = config.pop('num_constants', 0)
    layer = cls(cell, **config)
    layer._num_constants = num_constants
    return layer


  @property
  def _trackable_saved_model_saver(self):
    return layer_serialization.RNNSavedModelSaver(self)
