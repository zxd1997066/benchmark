import torch

from collections import namedtuple

from .cells import lstm_cell, premul_lstm_cell, flat_lstm_cell


# list[list[T]] -> list[T]
def flatten_list(lst):
    result = []
    for inner in lst:
        result.extend(inner)
    return result


'''
Define a creator as a function:
(options) -> (inputs, params, forward, backward_setup, backward)
inputs: the inputs to the returned 'forward'. One can call
    forward(*inputs) directly.
params: List[Tensor] all requires_grad=True parameters.
forward: function / graph executor / module
    One can call rnn(rnn_inputs) using the outputs of the creator.
backward_setup: backward_inputs = backward_setup(*outputs)
    Then, we pass backward_inputs to backward. If None, then it is assumed to
    be the identity function.
backward: Given `output = backward_setup(*forward(*inputs))`, performs
    backpropagation. If None, then nothing happens.

fastrnns.bench times the forward and backward invocations.
'''


ModelDef = namedtuple('ModelDef', [
    'inputs', 'params', 'forward', 'backward_setup', 'backward'])


def lstm_backward_setup(lstm_outputs, seed=None):
    hx, _ = lstm_outputs
    return simple_backward_setup(hx, seed)


def simple_backward_setup(output, seed=None):
    assert isinstance(output, torch.Tensor)
    if seed:
        torch.manual_seed(seed)
    grad_output = torch.randn_like(output)
    return output, grad_output


def simple_backward(output, grad_output):
    return output.backward(grad_output)


def pytorch_lstm_creator(**kwargs):
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    return ModelDef(
        inputs=[input, hidden],
        params=flatten_list(module.all_weights),
        forward=module,
        backward_setup=lstm_backward_setup,
        backward=simple_backward)


def lstm_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward)


def lstm_premul_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_premul(premul_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward)


def lstm_simple_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input] + [h[0] for h in hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_simple(flat_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward)


def lstm_multilayer_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden, flatten_list(params)]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_multilayer(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward)


def imagenet_cnn_creator(arch, jit=True):
    def creator(device='cuda', **kwargs):
        model = arch().to(device)
        x = torch.randn(32, 3, 224, 224, device=device)
        if jit:
            model = torch.jit.trace(model, x)
        return ModelDef(
            inputs=(x,),
            params=list(model.parameters()),
            forward=model,
            backward_setup=simple_backward_setup,
            backward=simple_backward)

    return creator


# input: lstm.all_weights format (wih, whh, bih, bhh = lstm.all_weights[layer])
# output: packed_weights with format
# packed_weights[0] is wih with size (layer, 4*hiddenSize, inputSize)
# packed_weights[1] is whh with size (layer, 4*hiddenSize, hiddenSize)
# packed_weights[2] is bih with size (layer, 4*hiddenSize)
# packed_weights[3] is bhh with size (layer, 4*hiddenSize)
def stack_weights(weights):
    def unzip_columns(mat):
        assert isinstance(mat, list)
        assert isinstance(mat[0], list)
        layers = len(mat)
        columns = len(mat[0])
        return [[mat[layer][col] for layer in range(layers)]
                for col in range(columns)]

    # XXX: script fns have problems indexing multidim lists, so we try to
    # avoid them by stacking tensors
    all_weights = weights
    packed_weights = [torch.stack(param)
                      for param in unzip_columns(all_weights)]
    return packed_weights


# returns: x, (hx, cx), all_weights, lstm module with all_weights as params
def lstm_inputs(seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
                miniBatch=64, return_module=False, device='cuda', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(seqLength, miniBatch, inputSize, device=device)
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers)
    if 'cuda' in device:
        lstm = lstm.cuda()

    if return_module:
        return x, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # wih, whh, bih, bhh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None


def lstm_factory(cell, script):
    def dynamic_rnn(input, hidden, wih, whh, bih, bhh):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden
        outputs = []
        inputs = input.unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn



# premul: we're going to premultiply the inputs & weights
def lstm_factory_premul(premul_cell, script):
    def dynamic_rnn(input, hidden, wih, whh, bih, bhh):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden
        outputs = []
        inputs = torch.matmul(input, wih.t()).unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), whh, bih, bhh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# simple: flat inputs (no tuples), no list to accumulate outputs
#         useful mostly for benchmarking older JIT versions
def lstm_factory_simple(cell, script):
    def dynamic_rnn(input, hx, cx, wih, whh, bih, bhh):
        hy = hx  # for scoping
        cy = cx  # for scoping
        inputs = input.unbind(0)
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], hy, cy, wih, whh, bih, bhh)
        return hy, cy

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def lstm_factory_multilayer(cell, script):
    def dynamic_rnn(input, hidden, params):
        # type: (Tensor, Tuple[Tensor, Tensor], List[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        params_stride = 4  # NB: this assumes that biases are there
        hx, cx = hidden
        hy, cy = hidden  # for scoping...
        inputs, outputs = input.unbind(0), []
        for layer in range(hx.size(0)):
            hy = hx[layer]
            cy = cx[layer]
            base_idx = layer * params_stride
            wih = params[base_idx]
            whh = params[base_idx + 1]
            bih = params[base_idx + 2]
            bhh = params[base_idx + 3]
            for seq_idx in range(len(inputs)):
                hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
                outputs += [hy]
            inputs, outputs = outputs, []
        return torch.stack(inputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn

