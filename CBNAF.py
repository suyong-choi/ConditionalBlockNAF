"""Implement Block NAF with conditional variables in Tensorflow
    following   https://github.com/johnpjust/UMA
    originally from pytorch https://github.com/nicola-decao/BNAF
    described in the paper https://doi.org/10.48550/arXiv.1904.04676

    Adapted by Suyong Choi (Korea University) to include conditional variables
"""
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow_probability as tfp

tfb = tfp.bijectors


def buildConditionalBNAF(inputdim, conddim, nafdim=32, depth=1, droprate=0.1, permute=True, gated=False):
    # first columns should be the inputdim the rest should be conditional variables

    xin = keras.layers.Input(shape=(inputdim+conddim, ))

    conditionalvar = xin[:, inputdim:]
    nhlayers = 3
    xint = xin
    for id in range(depth):
        
        if permute:
            randperm = np.random.permutation(inputdim).astype('int32')
            permutation = tf.constant(randperm, name=f'permutation{id}')
            #permutation = tf.Variable(randperm, name=f'permutation{idepth}', trainable=False)
            permuter = tfb.Permute(permutation=permutation, name=f'permute{id}')
            featurevar = xint[:, :inputdim]
            xfeatures_permuted = permuter.forward(featurevar)

            xint = tf.concat((xfeatures_permuted, conditionalvar), axis=1)
        # construct masked  layers
        xint = ConditionalMaskedweight(inputdim, 1, nafdim, conddim)(xint)
        xint = tf.nn.leaky_relu(xint)
        xint = keras.layers.Dropout(droprate)(xint) # drop out regularlization needed
        for _ in range(nhlayers):
            xint = ConditionalMaskedweight(inputdim, nafdim, nafdim)(xint)
            xint = tf.nn.leaky_relu(xint)
            xint = keras.layers.Dropout(droprate)(xint) # drop out regularlization needed
            #xint = keras.layers.BatchNormalization()(xint) # not sure about this
        xint = ConditionalMaskedweight(inputdim, nafdim, 1)(xint)
        
        # at the end unscramble
        if permute:
            xint = permuter.inverse(xint)
        
        # add some input to output if gated
        if gated:
            initializer = keras.initializers.RandomNormal()
            weightvar = tf.Variable(initial_value=tf.zeros(shape=(1,inputdim)), name=f'gatew{id}')
            alpha = tf.math.sigmoid(weightvar)
            xint = alpha * xint + (1.0-alpha) * xin[:, :inputdim]
        
        xterminal = xint
        xint = tf.concat( (xint, conditionalvar), axis=1)

    return keras.Model(xin, xterminal)


class ConditionalMaskedweight(keras.layers.Layer):

    def __init__(self, featinputdim : int, dperfeatin : int, dperfeatout : int, conddim:int=0):
        """_summary_

        Args:
            featinputdim (_type_): number of features to the flow
            dperfeatin (_type_): dimensions per feature on the input side
            dperfeatout (_type_): dimensions per feature on the output side
            conddim (int, optional): number of conditional features. Defaults to 0.
        """
        super(ConditionalMaskedweight, self).__init__()
        self.featinputdim = featinputdim # number of actual feature inputs (x space)
        self.conddim = conddim # number of conditional inputs
        self.dperfeatin = dperfeatin # number of input nodes per feature (in a given layer)
        self.dperfeatout = dperfeatout # number of output nodes per feature (in a given layer)
        self.inputdim = self.dperfeatin * self.featinputdim # total number of inputs
        self.outputdim = self.dperfeatout * self.featinputdim # totla number of outputs
        self.buildblockmatrix()
        pass

    def buildblockmatrix(self):
        """ Construct weight matrix whose size
        """

        # set up initial weights
        npweight = np.zeros(shape=(self.outputdim, self.inputdim), dtype=np.float32)
        # Construct Masks to retrieve diagonal and off-diagonal elements from weight later
        npmaskd = np.zeros_like(npweight, dtype=np.float32)
        npmasko = np.ones_like(npweight, dtype=np.float32)

        initializer = keras.initializers.GlorotUniform()

        for irow in range(self.featinputdim):
            rowstart = irow*self.dperfeatout
            rowend = (irow+1)*self.dperfeatout
            colstart = 0
            colend = (irow+1)*self.dperfeatin
            npweight[rowstart:rowend, colstart:colend] = \
                initializer(shape=(self.dperfeatout, colend)).numpy()

            npmaskd[rowstart:rowend, irow*self.dperfeatin:colend] = 1

            npmasko[rowstart:rowend, irow*self.dperfeatin:] = 0


        self.weight = tf.Variable(name='weight', initial_value=npweight)
        #biasinitval = tf.cast(tf.random_uniform(shape=(self.outputdim,), \
         #                   minval =  -1.0/np.sqrt(self.outputdim), \
          #                  maxval = 1.0/np.sqrt(self.outputdim)), dtype=tf.float32)
        #self.bias = tf.Variable("bias", initial_value=biasinitval)
        self.bias = tf.Variable(name = "bias", initial_value=initializer(shape=(self.outputdim,)))


        self.mask_d = tf.constant(name='mask_d', value=npmaskd, dtype=tf.float32)
        self.mask_o = tf.constant(name='mask_o', value=npmasko, dtype=tf.float32)

        if self.conddim > 0:
            self.condweight = tf.Variable(name="condweight", initial_value= initializer(shape=(self.outputdim, self.conddim)))
        else:
            self.condweight = tf.Variable(name="condweight", initial_value=0, trainable=False)
                                          
        #self._diag_weight = tf.Variable(name="diag_weight", \
        #                    initial_value=tf.math.log(tf.random.uniform(shape=(self.outputdim, 1), dtype=tf.float32)))
        self._diag_weight = tf.Variable(name="diag_weight", \
                            initial_value=tf.random.uniform(shape=(self.outputdim, 1), dtype=tf.float32))
        pass

    def get_weights(self, useweightnormalization=True):

        # If weight normalization is used, then not as sensitive to the choice
        # of exp or softplus
        #w = tf.multiply(tf.exp(self.weight), self.mask_d) + tf.multiply(self.weight, self.mask_o) 
        w = tf.multiply(tf.nn.softplus(self.weight), self.mask_d) + tf.multiply(self.weight, self.mask_o) 

        if self.conddim > 0:
            w = tf.concat( (w, self.condweight), 1)
        # use weight normalization from https://arxiv.org/abs/1602.07868
        if useweightnormalization:
            w_squared_norm = tf.reduce_sum(tf.math.square(w), axis=-1, keepdims=True)
            
            # using abs, exp is not good, softplus better
            #w = tf.math.exp(self._diag_weight) * w / tf.sqrt(w_squared_norm)
            w = tf.math.softplus(self._diag_weight) * w / tf.sqrt(w_squared_norm)

            wpl = self._diag_weight + self.weight - 0.5 * tf.math.log(w_squared_norm)
        else:
            wpl = self.weight

        wT = tf.transpose(w) # transpose of w. To be multiplied

        #wplmask = tf.reshape(tf.boolean_mask(wpl, tf.cast(self.mask_d, tf.bool)), (self.featinputdim, self.dperfeatout, self.dperfeatin) )


        return wT

    @tf.function
    def call(self, inputs: tf.Tensor, grad:tf.Tensor=None, **kwargs):
        wT = self.get_weights()

        affinetrans = tf.matmul(inputs, wT) + self.bias

        #if self.conddim>0:
        #    xcondin = inputs[:, self.inputdim:]
        #    affinetrans = tf.concat((affinetrans, xcondin), axis=1)
        return affinetrans
    
    def __repr__(self) -> str:
        return f'MaskedWeight(in_features={self.inputdim}, out_features={self.outputdim }, dim={self.featinputdim}, bias={not isinstance(self.bias, int)})'


if __name__ == '__main__':
    testmodel = buildConditionalBNAF(2, 4)