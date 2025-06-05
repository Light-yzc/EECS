"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU
import math

def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        x_pad = x.clone()
        pad =conv_param['pad']
        stride = conv_param['stride']
        if pad != 0:
            x_pad = torch.nn.functional.pad(x_pad, (pad,pad,pad,pad))
        x_shape = x.shape
        w_shape = w.shape
        out_shape_x = int(1 + (x_shape[2] + 2 * pad - w_shape[2]) / stride)
        out_shape_y = int(1 + (x_shape[3] + 2 * pad - w_shape[3]) / stride)
        out = torch.zeros(x.shape[0], w.shape[0], out_shape_x, out_shape_y,device=x.device,dtype=x.dtype)
        for n in range(x_shape[0]):
          out_tmp = torch.zeros(w_shape[0], out_shape_x, out_shape_y,device=x.device, dtype=x.dtype)
          for i in range(w_shape[0]):
              for x_i in range(out_shape_x):
                  for y_i in range(out_shape_y):
                    x_to_cal = x_pad[n,:,x_i * stride: x_i * stride + w_shape[2], y_i * stride: y_i * stride + w_shape[3]]
                    out_tmp[i, x_i, y_i] = torch.sum(x_to_cal * w[i]) + b[i]
          out[n] = out_tmp.unsqueeze(0)
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        x, w, b, conv_param = cache
        db = torch.zeros_like(b)
        db = torch.sum(dout, dim=(0,2,3),dtype=torch.float64)
        HH, WW = w.shape[2], w.shape[3]
        pad,stride =  conv_param['pad'], conv_param['stride']
        x_pad = torch.nn.functional.pad(x, (pad,pad,pad,pad))
        dw =torch.zeros_like(w) # (F, C, HH, WW)
        dxpad = torch.zeros_like(x_pad) # (N, C, H, W)
        F, N, H_out, W_out = w.shape[0], x.shape[0], dout.shape[2], dout.shape[3]
        for f_i  in range(F):
          for n_i in range(N):
              for h_out in range(H_out):
                  for w_out in range(W_out):
                    dout_now = dout[n_i, f_i, h_out, w_out]
                    cur_field = x_pad[n_i, : ,h_out * stride:h_out * stride + HH, w_out * stride: w_out *stride + WW]
                    dw[f_i] += cur_field * dout_now
                    dxpad[n_i, :, h_out * stride:h_out * stride + HH, w_out * stride:w_out * stride + WW] += dout_now * w[f_i]

        if pad != 0:
          dx = dxpad[:, :, pad:-pad, pad:-pad]
        else:
            dx =dxpad
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        pool_h = pool_param['pool_height']
        pool_w = pool_param['pool_width']
        stride = pool_param['stride']
        H_OUT = int(1 + (x.shape[2] - pool_h) / stride)
        W_OUT = int(1 + (x.shape[3] - pool_w) / stride)
        out = torch.zeros(size=(x.shape[0], x.shape[1], H_OUT, W_OUT),dtype=x.dtype, device=x.device)
        for n_i in range(x.shape[0]):
            for c_i in range(x.shape[1]):
                for h_i in range(H_OUT):
                    for w_i in range(W_OUT):
                        to_pool = x[n_i, c_i, h_i*stride:h_i*stride+pool_h, w_i*stride:w_i*stride+pool_w]
                        out[n_i, c_i, h_i, w_i] = torch.max(to_pool)
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x, pool_param = cache
        pool_h = pool_param['pool_height']
        pool_w = pool_param['pool_width']
        stride = pool_param['stride']
        dx = torch.zeros_like(x)
        H = dout.shape[2]
        W = dout.shape[3]
        for n_i in range(x.shape[0]):
            for c_i in range(x.shape[1]):
              for h in range(H):
                  for w in range(W):
                      # print(shape)
                        to_pool = x[n_i, c_i, h*stride:h*stride+pool_h, w*stride:w*stride+pool_w].clone()
                        _,index = torch.max(to_pool.reshape(-1), dim=0)
                        index_x = index // pool_h
                        index_y = index % pool_w
                        global_index_x = h*stride + index_x
                        global_index_y = w*stride + index_y
                        dx[n_i,c_i, global_index_x, global_index_y] = dout[n_i, c_i, h, w]
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        W1 = weight_scale * torch.randn(size=(num_filters, input_dims[0], filter_size, filter_size),device=device, dtype=dtype)
        b1 = torch.zeros(num_filters,device=device,dtype=dtype)
        pad = (filter_size - 1) // 2
        out_h = ((input_dims[1] - filter_size + 2 * pad) + 1 - 2) // 2 + 1
        W2 = weight_scale * torch.randn(size=(num_filters * out_h * out_h, hidden_dim),device=device,dtype=dtype)
        b2 = torch.zeros(hidden_dim,device=device,dtype=dtype)
        W3 = weight_scale * torch.randn(size=(hidden_dim, num_classes),device=device,dtype=dtype)
        b3 = torch.zeros(num_classes,device=device,dtype=dtype)
        self.params = {'W1':W1, 'W2':W2, 'W3':W3, 'b1':b1, 'b2':b2, 'b3':b3}
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        scores = 0
        cov1, cov_cache = FastConv.forward(X, W1, b1, conv_param)
        relu1, relu1_cache = ReLU.forward(cov1)
        pool1, pool_cache = FastMaxPool.forward(relu1, pool_param)
        line1, line1_cache = Linear.forward(pool1.reshape(X.shape[0], -1), W2, b2)
        relu2, relu2_cache = ReLU.forward(line1)
        line2, line2_cache = Linear.forward(relu2, W3, b3)
        scores = line2
        if y is None:
            return scores
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(line2, y)
        loss = loss + self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2) + torch.sum(W3 * W3))
        drelu2, dw3, db3 = Linear.backward(dout, line2_cache)
        dline1 = ReLU.backward(drelu2, relu2_cache)
        dpool, dw2, db2 = Linear.backward(dline1, line1_cache)
        dpool = dpool.reshape(pool1.shape) # to fold
        drelu1 = FastMaxPool.backward(dpool, pool_cache)
        dcov = ReLU.backward(drelu1, relu1_cache)
        dx, dw1, db1 = FastConv.backward(dcov, cov_cache)
        dw3 = dw3 + 2 * self.reg * W3
        dw2 = dw2 + 2 * self.reg * W2
        dw1 = dw1 + 2 * self.reg * W1
        grads = {'W1':dw1, 'W2':dw2, 'W3':dw3, 'b1':db1, 'b2':db2, 'b3':db3}
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        cur_w_dim = (input_dims[0],)
        out_dim = (input_dims[1], input_dims[2])
        for i, f in enumerate(num_filters):
            cur_w_dim = (f, cur_w_dim[0], 3, 3)
            if weight_scale == 'kaiming':
                w_tmp = kaiming_initializer(Din=cur_w_dim[1], Dout=cur_w_dim[0], K=cur_w_dim[2], relu=True, dtype=dtype, device=device)
            else:
                w_tmp = torch.randn(cur_w_dim,device=device, dtype=dtype) * weight_scale
            b_tmp = torch.zeros(f, device=device, dtype=dtype)
            self.params['W{}'.format(i+1)] = w_tmp
            self.params['b{}'.format(i+1)] = b_tmp
            out_h = 1 + (out_dim[0] + 2 * 1 - 3)
            out_dim = (out_h, out_h)
            if i in max_pools:
              out_h_pool = 1 + (out_h - 2) // 2
              out_dim = (out_h_pool, out_h_pool)
            if self.batchnorm:
                self.params['gamma{}'.format(i+1)] = 0.01 * torch.randn(w_tmp.shape[0], dtype=dtype, device=device)
                self.params['beta{}'.format(i+1)] = 0.01 * torch.randn(w_tmp.shape[0],dtype=dtype, device=device)
        if weight_scale == 'kaiming':
            self.params['W{}'.format(i+2)] = kaiming_initializer(f * out_dim[0] * out_dim[1],num_classes,relu= False,dtype=dtype, device=device)
        else:
            self.params['W{}'.format(i+2)] = torch.randn(f * out_dim[0] * out_dim[1], num_classes, device=device, dtype=dtype) * weight_scale
        self.params['b{}'.format(i+2)] = torch.zeros(num_classes, device=device, dtype=dtype)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        cur_x = X
        cache_param = []
        if self.batchnorm:
            conv_pool = Conv_BatchNorm_ReLU_Pool()
            conv = Conv_BatchNorm_ReLU()
        else:
            conv_pool = Conv_ReLU_Pool()
            conv = Conv_ReLU()
        for i in range(self.num_layers - 1):
            if i in self.max_pools:
              if not self.batchnorm:
                  cur_out, cur_cache = conv_pool.forward(cur_x, self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)], conv_param, pool_param)
              else:
                  # print(cur_x.shape)
                  # print(self.params['W{}'.format(i+1)].shape, self.params['b{}'.format(i+1)].shape, self.params['gamma{}'.format(i+1)].shape, self.params['beta{}'.format(i+1)].shape)
                  cur_out, cur_cache = conv_pool.forward(cur_x, self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)], self.params['gamma{}'.format(i+1)], self.params['beta{}'.format(i+1)],conv_param,self.bn_params[i], pool_param)
            else:
              if not self.batchnorm:
                  cur_out, cur_cache = conv.forward(cur_x, self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)], conv_param)
              else:
                  cur_out, cur_cache = conv.forward(cur_x, self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)], self.params['gamma{}'.format(i+1)],self.params['beta{}'.format(i+1)],conv_param,self.bn_params[i])
            cur_x = cur_out
            cache_param.append(cur_cache)
        scores, cur_cache = Linear.forward(cur_x.reshape(X.shape[0], -1), self.params['W{}'.format(i+2)], self.params['b{}'.format(i+2)])
        cache_param.append(cur_cache)
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        l2 = 0
        loss, dout = softmax_loss(scores, y)
        for i in range(self.num_layers):
            l2 += torch.sum(self.params['W{}'.format(i+1)] ** 2)
        loss += l2 * self.reg
        for i in range(self.num_layers, 0, -1):
          if i == self.num_layers:
              d_x_tmp, dw_tmp, db_tmp = Linear.backward(dout, cache_param[i - 1])
              d_x_tmp = d_x_tmp.reshape(cur_x.shape)
          else:
              if not self.batchnorm:
                  if i-1 in self.max_pools:
                      d_x_tmp, dw_tmp, db_tmp = conv_pool.backward(dout, cache_param[i - 1])
                  else:
                      d_x_tmp, dw_tmp, db_tmp = conv.backward(dout, cache_param[i - 1])
              else:
                  # print('pooled')
                    if i-1 in self.max_pools:
                        d_x_tmp, dw_tmp, db_tmp, dgamma_tmp, dbeta_tmp = conv_pool.backward(dout, cache_param[i - 1])
                    else:
                        d_x_tmp ,dw_tmp, db_tmp , dgamma_tmp, dbeta_tmp = conv.backward(dout, cache_param[i - 1])
                    grads['gamma{}'.format(i)] = dgamma_tmp # + 2 * self.reg * self.params['gamma{}'.format(i)]
                    grads['beta{}'.format(i)] = dbeta_tmp
          grads['W{}'.format(i)] = dw_tmp + 2 * self.reg * self.params['W{}'.format(i)]
          grads['b{}'.format(i)] = db_tmp
          dout = d_x_tmp
          # print('to_i:{} cur_i:{} max:{}'.format(self.num_layers, i, torch.max(d_x_tmp)))
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################
        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-1   # Experiment with this!
    learning_rate = 1e-2  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    X_train = data_dict['X_train']
    model = DeepConvNet(
        input_dims=X_train.shape[1:], num_classes=10,
        num_filters=[32, 16, 64],
        max_pools=[0,1,2],
        weight_scale='kaiming',
        reg=1e-4,
        dtype=dtype,
        device=device
    )
    solver = Solver(
        model, data_dict, num_epochs = 100,
        batch_size = 512,
        update_rule = adam,
        optim_config={
            'learning_rate':1.5e-3
        },
        print_every=20,device=device   
    )

    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        weight_scale = math.sqrt(gain / Din)
        weight = torch.randn(Din, Dout, dtype=dtype ,device=device) * weight_scale
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        fan_in = Din * K * K
        weight_scale = math.sqrt(gain / fan_in)
        weight = torch.randn(Dout, Din, K, K , dtype=dtype ,device=device) * weight_scale
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            mean = torch.mean(x, dim=0)
            var =  torch.mean((x - mean) ** 2, dim=0)
            x_hat = (x - mean) / torch.sqrt(var + eps)
            out = gamma * x_hat + beta
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var
            cache = (x, gamma, beta, x_hat, mean, var, eps)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
            out = x_hat * gamma + beta 
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, 
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        x, gamma, beta, x_norm, sample_mean, sample_var, eps = cache
        N, D = dout.shape

        # 1. 计算 dbeta
        dbeta = torch.sum(dout, dim=0)

        # 2. 计算 dgamma
        dgamma = torch.sum(dout * x_norm, dim=0)

        # 3. 计算 dx_norm
        dx_norm = dout * gamma

        # 4. 计算 dvar
        # 这一项是 dout * gamma * (x - mu) * (-0.5) * (var + eps)^(-3/2)
        dvar = torch.sum(dx_norm * (x - sample_mean), dim=0) * (-0.5) * torch.pow(sample_var + eps, -1.5)

        # 5. 计算 dmu (dmean)
        # 这一项是 dout * gamma * (-1/sqrt(var+eps)) + dvar * (-2 * (x - mu)) / N
        dmu = torch.sum(dx_norm * (-1 / torch.sqrt(sample_var + eps)), dim=0) + \
              dvar * torch.sum(-2 * (x - sample_mean), dim=0) / N
        # 或者更简洁地:
        # dmu = torch.sum(dx_norm / (-torch.sqrt(sample_var + eps)), dim=0)

        # 6. 计算 dx
        # dx = dx_norm / sqrt(var + eps) + dvar * 2 * (x - mu) / N + dmu / N
        dx = dx_norm / torch.sqrt(sample_var + eps) + \
            dvar * 2 * (x - sample_mean) / N + \
            dmu / N
        # N,D = dout.shape
        # dbeta = torch.sum(dout, dim=0)
        # dgamma = torch.sum((x_hat * dout), dim=0)
        # dx_hat = dout * gamma
        
        # # 3. 计算 std = sqrt(var + eps)
        # std = torch.sqrt(var + eps)
        
        # # 4. 计算两个求和项
        # sum_dx_hat = torch.sum(dx_hat, dim=0)          # 形状 (D,)
        # sum_dx_hat_x_hat = torch.sum(dx_hat * x_hat, dim=0)  # 形状 (D,)
        
        # # 5. 使用向量化公式计算 dx
        # dx = (1.0 / (N * std)) * (N * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat)
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape[0],x.shape[1]
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))
        x_back = x.permute(0,2,3,1).reshape(-1, x.shape[1])
        if mode == 'train':
            mu = torch.mean(x_back, dim=0)
            var = torch.mean((x_back - mu) ** 2, dim= 0)
            # var = torch.var(x_back, dim=0)
            form = (x_back - mu) / torch.sqrt(var +  eps)
            out_tmp = form * gamma + beta
            out = out_tmp.reshape(x.permute(0,2,3,1).shape).permute(0, 3, 1, 2)
            # print(running_mean.shape, mu.shape, x_back.shape)
            # print(bn_param)
            running_mean = momentum * running_mean + (1 - momentum) * mu
            running_var = momentum * running_var + (1 - momentum) * var
            bn_param['running_mean'] = running_mean
            bn_param['running_var'] = running_var
            cache = (x_back, gamma, beta, form, mu, var, eps)

        elif mode == 'test':
            form = (x_back - running_mean) / torch.sqrt(running_var + eps)
            out_tmp = form * gamma + beta
            out = out_tmp.reshape(x.permute(0,2,3,1).shape).permute(0, 3, 1, 2)
        ###############################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        # dout(N, C, H, W) -> (N * H * W, C)
        # dx_tmp （N * H * W, C) -> (N, C, H ,W)
        dout_reshape = dout.permute(0, 2, 3, 1).reshape(-1,dout.shape[1])
        N,C,H,W = dout.shape
        dx_tmp, dgamma, dbeta = BatchNorm.backward(dout_reshape, cache)
        dx =dx_tmp.reshape(N, H, W, C).permute(0, 3, 1, 2)
        # N = dout_reshape.shape[0]
        # dbeta = torch.sum(dout_reshape, dim=0)
        # dgamma = torch.sum(x_hat * dout_reshape, dim=0)
        # dx_hat = dout_reshape * gamma
        # std = torch.sqrt(var + eps)
        # sum_dx_hat = torch.sum(dx_hat, dim=0)
        # sum_hat_hat_dx = torch.sum(dx_hat * x_hat, dim=0)  # dx = (1 / N) * (N - sum_x_hat - x_hat * sum(x_hat * x_hat))
        # dx = ((1.0 / (N * std)) * (N * dx_hat - sum_dx_hat - x_hat * sum_hat_hat_dx)).reshape(dout.shape).permute(0,1,2,3)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
