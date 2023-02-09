# utils.py

"""
This file defines useful utility functions for the dna pred model.

There are two sections:

1.  Creating a version of the Enformer model suitable for transfer
    learning by chopping off the subsequent layers after attention
    and passing the weights into a newly created enformer model 
    (Tensorflow).
2.  Utility for Enformer Celltyping model.  
    2.1 Set up and train the model
    2.2 Inspect & using model
    
"""


# SECTION 1 TRANSFER LEARNING -------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys

from tensorflow.core.protobuf import graph_debug_info_pb2
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.load import Loader
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

import tensorflow as tf
#get enformer initialised
import sonnet as snt
from EnformerCelltyping.attention_module import *
#enf model converted to keras
from EnformerCelltyping.enformer import Enformer
from EnformerCelltyping.enformer import TargetLengthCrop1D
from EnformerCelltyping.enformer_chopped import Enformer_Chopped

def _get_loader(export_dir, tags=None, options=None):
    """
    Loader implementation.

    Custom to get weights from tf.hub model.

    Created by William Beardall <william.beardall15@imperial.ac.uk>
    """
    options = options or load_options.LoadOptions()
    if tags is not None and not isinstance(tags, set):
      # Supports e.g. tags=SERVING and tags=[SERVING]. Sets aren't considered
      # sequences for nest.flatten, so we put those through as-is.
      tags = nest.flatten(tags)
    saved_model_proto, debug_info = (
        loader_impl.parse_saved_model_with_debug_info(export_dir))

    if (len(saved_model_proto.meta_graphs) == 1 and
        saved_model_proto.meta_graphs[0].HasField("object_graph_def")):
      meta_graph_def = saved_model_proto.meta_graphs[0]
      # tensor_content field contains raw bytes in litle endian format
      # which causes problems when loaded on big-endian systems
      # requiring byteswap
      if sys.byteorder == "big":
        saved_model_utils.swap_function_tensor_content(meta_graph_def, "little",
                                                       "big")
      if (tags is not None
          and set(tags) != set(meta_graph_def.meta_info_def.tags)):
        raise ValueError(
            ("The SavedModel at {} has one MetaGraph with tags {}, but got an "
             "incompatible argument tags={} to tf.saved_model.load. You may omit "
             "it, pass 'None', or pass matching tags.")
            .format(export_dir, meta_graph_def.meta_info_def.tags, tags))
      object_graph_proto = meta_graph_def.object_graph_def

      ckpt_options = checkpoint_options.CheckpointOptions(
          experimental_io_device=options.experimental_io_device)
      with ops.init_scope():
        try:
          loader = Loader(object_graph_proto, saved_model_proto, export_dir,
                              ckpt_options, None)
        except errors.NotFoundError as err:
          raise FileNotFoundError(
              str(err) + "\n If trying to load on a different device from the "
              "computational device, consider using setting the "
              "`experimental_io_device` option on tf.saved_model.LoadOptions "
              "to the io_device such as '/job:localhost'."
          )
        return loader

def create_enf_model(path: str,test_against_tf_hub: bool = True):
    """
    Lift weights from pre-trained Enformer model available in tf.hub

    and move to tf.keras model so that it can be used for fine-tuning.

    Includes tests to ensure model performs _close to_ pre-trained 
    
    model's predictions.
    
    Requires 16GB+ of RAM to run
    
    Arguements:
    path: Path to enformer model scripts.
    test_against_tf_hub: whether to test accuracy of recreated model 
    against tf.hub model. It is advised to do this but requires internet
    access.
    """
    
    #loader = _get_loader(str(DATA_PATH / "enformer_model"))
    loader = _get_loader(path)
    variable_nodes = [n for n in loader._nodes if isinstance(n,tf.Variable)]
    #first one isn't a tf.Variable from the model, remove it
    variable_nodes = variable_nodes[1:]

    #construct an untrained model with the aim to add the learned weights
    #create model with initialised weights
    model = Enformer(channels=1536,
                     num_heads=8,
                     num_transformer_layers=11,
                     pooling_type='attention')
    
    #have to pass data through model to initialise all the weights
    #note this will require quite a bit of RAM
    #outputs = model(tf.zeros([1,196_608,4], tf.float32), is_training=True)
    #{'sequence': (TensorShape([1, 131072, 4]), tf.float32), 'target': (TensorShape([1, 896, 5313]), tf.float32)}
    g1 = tf.random.Generator.from_seed(1)
    rand_batch = {'sequence': tf.zeros([1,196_608,4], tf.float32),'target': g1.normal(shape=[1, 896, 5313])}
    output = model(rand_batch['sequence'], is_training=True)

    #now let's make sure the number of weights is the same
    init_enf_names = [n.name for n in model.variables]
    extrac_weight_names = [n.name for n in variable_nodes]
    
    assert len(extrac_weight_names)==len(init_enf_names), "Number of weights don't match"

    #order and naming needs to be updated, once they match we can update by name
    for i in range(len(variable_nodes)):
        variable_nodes[i]=tf.Variable(variable_nodes[i].value(),name='enformer'+extrac_weight_names[i][5:])
    #update names
    extrac_weight_names = [n.name for n in variable_nodes]

    #model duplicates the naming convention after second '/', do the same
    to_dup = ['/final_pointwise/','/stem/','/conv_tower/','/transformer/','/head_human/',
              '/head_mouse/','/mha/','_layer/linear/','/normalization/','/batch_norm/layer_norm/',
              '/conv_block/','/pointwise_conv_block/',
              '/downres/','/cross_replica_batch_norm/',
              '/exponential_moving_average/',
              '/normalization/layer_norm/',
              '/pooling/softmax_pooling/',
              '/conv_block/conv_block/batch_norm/batch_norm/',
              '/pointwise_conv_block/pointwise_conv_block/batch_norm/batch_norm/',
              'enformer/', #don't do this when passing snt.Model
              '/mlp/'] + ['/transformer_block_'+str(i)+'/'
                          for i in range(11)]+['/downres_block_'+str(i)+'/' for i in range(11)]
    dup_with = ['/final_pointwise/final_pointwise/','/stem/stem/','/conv_tower/conv_tower/',
                '/transformer/transformer/','/head_human/head_human/',
                '/head_mouse/head_mouse/','/mha/mha/','_layer/','/batch_norm/','/layer_norm/',
                '/conv_block/conv_block/','/pointwise_conv_block/pointwise_conv_block/',
                '/conv_tower/conv_tower/','/batch_norm/',
                '/moving_mean/',
                '/layer_norm/',
                '/softmax_pooling/',
                '/conv_block/conv_block/batch_norm/',
                '/pointwise_conv_block/pointwise_conv_block/batch_norm/',
                '',
                '/mlp/mlp/'] + ['/transformer_block_'+str(i)+'/transformer_block_'+str(i)+'/'
                                for i in range(11)]+['/conv_tower_block_'+str(i)+'/conv_tower_block_'+str(i)+'/'
                                                     for i in range(11)]
    #also remove ending of :0:0 with :0 - happens when you update the name
    end_long = ':0:0'
    end_short = ''
    
    for i in range(len(variable_nodes)):
        new_name = variable_nodes[i].name
        for j,dup_name in enumerate(to_dup):
            if(dup_name in new_name):
                new_name = new_name.replace(dup_name,dup_with[j])
                #attention_k needs to match transformer_block_k
                if('/transformer_block_' in dup_name):
                    #also replace...
                    if('/multihead_attention/' in new_name):
                        new_name = new_name.replace('/multihead_attention/',
                                                    '/attention_'+dup_name.rsplit('_', 1)[1])


        new_name = new_name.replace(end_long,end_short)
        variable_nodes[i]=tf.Variable(variable_nodes[i].value(),name=new_name)

    #update names
    extrac_weight_names = [n.name for n in variable_nodes]

    #add in shape too since names aren't unique in either
    extrac_weight_names = [n.name+' : '+str(n.shape) for n in variable_nodes]
    init_enf_names = [n.name+' : '+str(n.shape) for n in model.variables]

    assert len(list(set(extrac_weight_names) - set(init_enf_names)))==0, "Not all extracted weights are in initialised"

        #issue is moving_variance is missing from the extracted weights and is named moving_mean instead
    #can't tell which ones should be variance and which mean as shape is the same so let's just guess and test
    #Approach: rename the second set of moving_mean to moving_variance (in chronological order)
    extrac_weight_names = [n.name for n in variable_nodes]
    init_enf_names = [n.name for n in model.variables]
    
    #find those to be updated
    #[n.name+' : '+str(i) for i,n in enumerate(variable_nodes)]
    indices_change_var = [36,37,38,42,43,44,138,139,140,146,147,148,154,155,156,162,163,164,170,171,
                          172,178,179,180,230,231,232,236,237,238,242,243,244,248,249,250,254,255,256,
                          260,261,262
                         ]
    #also remove ending of :0:0 with :0 - happens when you update the name
    end_long = ':0'
    end_short = ''
    for i in indices_change_var:
        new_name = variable_nodes[i].name
        new_name = new_name.replace('moving_mean','moving_variance')
        new_name = new_name.replace(end_long,end_short)
        variable_nodes[i]=tf.Variable(variable_nodes[i].value(),name=new_name)

    #update names
    extrac_weight_names = [n.name for n in variable_nodes]
    
    #check they are matching
    extrac_weight_names = [n.name+' : '+str(n.shape) for n in variable_nodes]
    init_enf_names = [n.name+' : '+str(n.shape) for n in model.variables]

    assert len(list(set(extrac_weight_names) - set(init_enf_names)))==0, "Not all extracted weights are in initialised"
    assert len(list(set(init_enf_names) - set(extrac_weight_names)))==0, "Not all initialised weights are in extracted"

    #great so now we can update the initiated model with the weights from the pre-trained
    #get index where there is a match
    for init_i, weight_name in enumerate(init_enf_names):
        extrac_i = extrac_weight_names.index(weight_name)
        #update weight values
        model.variables[init_i].assign(variable_nodes[extrac_i])


    #now let's predict with the two models to check that the weights have been correctly assigned
    #tf.hub model takes in extra seq either side and then just filters - no good reason why it does this
    zeros = tf.zeros([1,196_608,4], tf.float32)
    zeros_long = tf.zeros([1,393_216,4], tf.float32)

    #Enformer from tf hub
    if (test_against_tf_hub):
        import tensorflow_hub as hub
        enformer_model_hub = hub.load("https://tfhub.dev/deepmind/enformer/1").model
        enformer_model_hub_preds = enformer_model_hub.predict_on_batch(zeros_long)
        #created Enformer
        model_preds = model(zeros, is_training=False)

        #check that difference is negligible
        mouse_diff = tf.reduce_mean(tf.keras.losses.mean_squared_error(enformer_model_hub_preds['mouse'], 
                                                                       model_preds['mouse']))
        human_diff = tf.reduce_mean(tf.keras.losses.mean_squared_error(enformer_model_hub_preds['human'], 
                                                                       model_preds['human']))
        assert mouse_diff<1e-06,"MSE difference in mouse predictions is above permissible threshold"
        assert human_diff<1e-06,"MSE difference in human predictions is above permissible threshold"

        #check on random sequence
        np.random.seed(42)
        EXTENDED_SEQ_LENGTH = 393_216
        SEQ_LENGTH = 196_608
        inputs = np.array(np.random.random((1, EXTENDED_SEQ_LENGTH, 4)), dtype=np.float32)
        inputs_cropped = TargetLengthCrop1D(SEQ_LENGTH)(inputs)

        enformer_model_hub_preds = enformer_model_hub.predict_on_batch(inputs)
        #created Enformer
        model_preds = model(inputs_cropped, is_training=False)
        #check that difference is negligible
        mouse_diff = tf.reduce_mean(tf.keras.losses.mean_squared_error(enformer_model_hub_preds['mouse'], 
                                                                       model_preds['mouse']))
        human_diff = tf.reduce_mean(tf.keras.losses.mean_squared_error(enformer_model_hub_preds['human'], 
                                                                       model_preds['human']))

        assert mouse_diff<1e-06,"MSE difference in mouse predictions is above permissible threshold"
        assert human_diff<1e-06,"MSE difference in human predictions is above permissible threshold"
        #Enformer now has a method to load weights too but seems ever so slightly worse on this test:
        #mouse_diff (this version) 3.2464435e-08 (enformer's version) 5.4844513e-08
        #human_diff (this version) 1.9401652e-07 (enformer's version) 1.14528476e-07
        #really they are pretty much the same
    
    
    #Finally return the model so it can be used for fine-tuning
    return(model)

#tf sonnet model
def create_enf_chopped_model(path: str):
    """
    Lift weights from pre-trained Enformer model available in tf.hub
    and move to chopped version of Enformer model which has below 
    architecture. This can then be used in a fine-tuning approach.
    Keep architecture:
    
    1. Stem 
    2. Conv Tower 
    3. Transformer with relative positional encodings
    4. Crop. 
    
    Requires >24GB of RAM to run
    
    Arguements:
    
    path: 
        Path to enformer model scripts.
    """
    
    loader = _get_loader(path)
    variable_nodes = [n for n in loader._nodes if isinstance(n,tf.Variable)]
    #first one isn't a tf.Variable from the model, remove it
    variable_nodes = variable_nodes[1:]
    
    #construct an untrained chopped enformer model with the aim to add the 
    #learned weights 
    #create model with initialised weights    
    enf_chopped = Enformer_Chopped(channels=1536,
                                   num_heads=8,
                                   num_transformer_layers=11,
                                   pooling_type='attention')
    #have to pass data through model to initialise all the weights
    #note this will require quite a bit of RAM
    g1 = tf.random.Generator.from_seed(1)
    input_= tf.zeros([1,196_608,4], tf.float32)
    rand_batch = {'sequence': input_,'target': g1.normal(shape=[1, 896, 6])}
    output = enf_chopped(rand_batch['sequence'], is_training=True)

    #now let's make sure the number of weights is the same
    init_enf_names = [n.name for n in enf_chopped.variables]
    extrac_weight_names = [n.name for n in variable_nodes]
    #Number of weights won't match, we only want to add weights for certain bits
    #order and naming needs to be updated, once they match we can update by name
    for i in range(len(variable_nodes)):
        variable_nodes[i]=tf.Variable(variable_nodes[i].value(),
                                      name='enformer_chopped'+extrac_weight_names[i][5:])
    #update names
    extrac_weight_names = [n.name for n in variable_nodes]
    
    #also remove ending of :0:0 with :0 - happens when you update the name
    end_long = ':0:0'
    end_short = ''

    #model duplicates the naming convention after second '/', do the same
    to_dup = ['/stem/','/conv_tower/','/transformer/',
              '/mha/','/conv_block/','/conv_tower/',
              '/mlp/','/pointwise_conv_block/',
              '_layer/linear/','/normalization/','/batch_norm/layer_norm/',
              '/downres/','/cross_replica_batch_norm/',
              '/exponential_moving_average/',
              '/normalization/layer_norm/',
              '/pooling/softmax_pooling/',
              '/conv_block/conv_block/batch_norm/batch_norm/',
              '/pointwise_conv_block/pointwise_conv_block/batch_norm/batch_norm/',
              'conv_block/batch_norm/batch_norm/',
    ] + ['/transformer_block_'+str(i)+'/'
         for i in range(11)]+['/downres_block_'+str(i)+'/' for i in range(11)]
    dup_with = ['/stem/stem/','/conv_tower/conv_tower/','/transformer/transformer/',
                '/mha/mha/','/conv_block/conv_block/','/conv_tower/conv_tower/',
                '/mlp/mlp/','/pointwise_conv_block/pointwise_conv_block/',
                '_layer/','/batch_norm/','/layer_norm/',
                '/conv_tower/conv_tower/','/batch_norm/',
                '/moving_mean/',
                '/layer_norm/',
                '/softmax_pooling/',
                '/conv_block/conv_block/batch_norm/',
                '/pointwise_conv_block/pointwise_conv_block/batch_norm/',
                'conv_block/batch_norm/',
    ] + ['/transformer_block_'+str(i)+'/transformer_block_'+str(i)+'/'
                                for i in range(11)]+['/conv_tower_block_'+str(i)+'/conv_tower_block_'+str(i)+'/'
                                                     for i in range(11)]
        
    for i in range(len(variable_nodes)):
        new_name = variable_nodes[i].name
        for j,dup_name in enumerate(to_dup):
            if(dup_name in new_name):
                new_name = new_name.replace(dup_name,dup_with[j])
                #attention_k needs to match transformer_block_k
                if('/transformer_block_' in dup_name):
                    #also replace...
                    if('/multihead_attention/' in new_name):
                        new_name = new_name.replace('/multihead_attention/',
                                                    '/attention_'+dup_name.rsplit('_', 1)[1])


        new_name = new_name.replace(end_long,end_short)
        variable_nodes[i]=tf.Variable(variable_nodes[i].value(),name=new_name)    

    #update names
    extrac_weight_names = [n.name for n in variable_nodes]
    
    #Note not all extracted weights will be in initialised since layers renamed

    #issue is moving_variance is named moving_mean in the extracted weights 
    #can't tell which ones should be variance and which mean as shape is the same so 
    #using order to tell
    #Approach: rename the second set of moving_mean to moving_variance (in chronological order)
    extrac_weight_names = [n.name for n in variable_nodes]
    init_enf_names = [n.name for n in enf_chopped.variables]
    
    #find those to be updated
    #[n.name+' : '+str(i) for i,n in enumerate(variable_nodes)]
    indices_change_var = [36,37,38,42,43,44,138,139,140,146,147,148,154,155,156,162,163,164,170,171,
                          172,178,179,180,230,231,232,236,237,238,242,243,244,248,249,250,254,255,256,
                          260,261,262
                         ]
    #also remove ending of :0:0 with :0 - happens when you update the name
    end_long = ':0'
    end_short = ''
    for i in indices_change_var:
        new_name = variable_nodes[i].name
        #variance only there in keras model not snt
        new_name = new_name.replace('moving_mean','moving_variance')
        new_name = new_name.replace(end_long,end_short)
        variable_nodes[i]=tf.Variable(variable_nodes[i].value(),name=new_name)

    #update names
    extrac_weight_names = [n.name for n in variable_nodes]
    
    #check they are matching
    extrac_weight_names = [n.name+' : '+str(n.shape) for n in variable_nodes]
    init_enf_names = [n.name+' : '+str(n.shape) for n in enf_chopped.variables]
    
    #we only want to update the matching layers in the trunk section
    extrac_trunk_weight_names = [
        layer for layer in extrac_weight_names if ('/trunk/' in layer and 
                                                   '/final_pointwise/' not in layer)]

    assert len(list(set(extrac_trunk_weight_names) - 
                    set(init_enf_names)))==0, "Not all extracted weights are in initialised"
    
    #get names of model.trainable_variables so can return index
    init_enf_train_names = [n.name+' : '+str(n.shape) for n in enf_chopped.trainable_variables]
    
    #great so now we can update the initiated model with the weights from the pre-trained
    # for 1. Stem 2. Conv Tower 3. Transformer 4. Crop of trunk 
    #get index where there is a match
    trainable_variables = []
    for init_i, variable in enumerate(enf_chopped.variables):
        weight_name = variable.name + ' : ' + str(variable.shape)
        #make sure only updating trunk weights from 1-4 above
        if (weight_name in extrac_trunk_weight_names):
            extrac_i = extrac_weight_names.index(weight_name)
            variable_values = variable_nodes[extrac_i]
            #update weight values
            enf_chopped.variables[init_i].assign(variable_values)
        #if freezing pre-trained layers, return other layers' indices
        elif(weight_name in init_enf_train_names):
            train_init_i = init_enf_train_names.index(weight_name)
            trainable_variables.append(train_init_i)     
            
    return(enf_chopped)




# SECTION 2 Utility for Enformer Celltyping -----------------------------------------------
 
# 2.1 Set up and train the model --------------------------------------------


#gelu layer, from Enformer -----------------------------------------
def gelu(x: tf.Tensor,name=None) -> tf.Tensor:
  """
  Applies the Gaussian error linear unit (GELU) activation function.
  Using approximiation in section 2 of the original paper:
  https://arxiv.org/abs/1606.08415
  Args:
    x: Input tensor to apply gelu activation.
    name: Name of the layer
  Returns:
    Tensor with gelu activation applied to it.
  """
  return tf.nn.sigmoid(1.702 * x,name=name) * x

#Pearson R Implementation from Enformer-----------------------------
@tf.function  
def pearsonR(y_true, y_pred,reduce_axis=(0,1),avg=True):
    _product_sum = tf.reduce_sum(y_true * y_pred, axis=reduce_axis)
    _true_sum = tf.reduce_sum(y_true, axis=reduce_axis)
    _pred_sum = tf.reduce_sum(y_pred, axis=reduce_axis)
    _count = tf.reduce_sum(tf.ones_like(y_true), axis=reduce_axis)
    
    _true_squared_sum = tf.reduce_sum(tf.math.square(y_true), 
                                      axis=reduce_axis)
    _pred_squared_sum = tf.reduce_sum(tf.math.square(y_pred), 
                                      axis=reduce_axis)
    
    true_mean = _true_sum / _count
    pred_mean = _pred_sum / _count
    
    covariance = (_product_sum
                  - true_mean * _pred_sum
                  - pred_mean * _true_sum
                  + _count * true_mean * pred_mean)
    
    true_var = _true_squared_sum - _count * tf.math.square(true_mean)
    pred_var = _pred_squared_sum - _count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)

    correlation = covariance / tp_var
    #now just get mean of all channels
    if avg:
        #remove nans caused by 0's in a true or pred channel
        return tf.reduce_mean(tf.boolean_mask(correlation, tf.math.is_finite(correlation)))
    return correlation

#mulit-channel MSE, split by channel--------------------------------
from tensorflow.keras.metrics import mean_squared_error

#@tf.function
def multi_mse(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)

#@tf.function
def multi_mse_for_class(index,num_s,num_l):
    def multi_mse_inner(true,pred):
        #get indexs of hist mark
        indexs = list(np.arange(index,num_s*num_l,num_l))
        #get only the desired class
        true = tf.gather(true,indexs,axis=2)
        pred = tf.gather(pred,indexs,axis=2)
        #return dice per class
        return multi_mse(true,pred)
    #have to give each a unique name or metrics call will give out
    multi_mse_inner.__name__='multi_mse_inner'+str(index)
    return multi_mse_inner


#Data loading, training, test split functions -----------------------

import os
import pathlib
from typing import Sequence, Union

import numpy as np
import pandas as pd
import random
import pyBigWig

import itertools
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

from EnformerCelltyping.constants import (
    CHROMOSOMES,
    CELLS,
    ALLOWED_CELLS,
    ALLOWED_FEATURES,
    CHROMOSOME_DATA,
    DNA_DATA,
    BLACKLIST_PATH,
    ATAC_DATA,
    H3K4ME1_DATA,
    H3K27AC_DATA,
    H3K4ME3_DATA,
    H3K9ME3_DATA,
    H3K27ME3_DATA,
    H3K36ME3_DATA,
    MODEL_REFERENCE_PATH,
    METADATA_PATH,
    AVG_DATA_PATH
)


def train_valid_split(chromosomes, chrom_len, samples, valid_frac, split):
    """
    Function to create a train validatin split between chromosomes.
    Takes list of chromosomes, list of their lengths, the list of
    sample types and the fraction dedicated to validation data.
    Split type is either CHROM, SAMPLE or BOTH and
    determines if specific chromosomes are used for
    train/test or if its split by samples or if both are used
    """

    def full_index_dist(n,chrom_len=None):
        index = np.asarray([x for x in range(0, n)])
        dist = np.ones(n)/len(index) #proportions
        if chrom_len is not None:
            dist = chrom_len/sum(chrom_len)
        return index, dist

    def sample_index_dist(n, frac, chrom_len=None):
        """
        If chr (chromosome is true) then use the len of each 
        chromosome in the selection dist
        """
        tr_count = int(n*(1-frac))
        tr_index = random.sample(range(0, n), tr_count)
        ts_index = [x for x in range(0, n) if x not in tr_index]
        tr_dist = np.ones(tr_count)/tr_count
        ts_dist = np.ones(n-tr_count)/(n-tr_count)
        if chrom_len is not None:
            tr_dist = chrom_len[tr_index]/sum(chrom_len[tr_index])
            ts_dist = chrom_len[ts_index]/sum(chrom_len[ts_index])
        return tr_index, ts_index, tr_dist, ts_dist

    # RUN PIECE OF CODE BELOW BUT CHANGE CHROMOSOMES TO THE RELEVANT
    # TYPE OF DATA. ALSO REPROGRAM dist selection depending on
    # split type.
    if split == 'CHROM':
        (c_train_index, c_valid_index,
         c_train_dist, c_valid_dist) = sample_index_dist(len(chromosomes),
                                                        valid_frac, chrom_len)
        s_train_index, s_train_dist = full_index_dist(len(samples))
        s_valid_index, s_valid_dist = full_index_dist(len(samples))
    if split == 'SAMPLE':
        (s_train_index, s_valid_index,
         s_train_dist, s_valid_dist) = sample_index_dist(len(samples),
                                                        valid_frac)
        c_train_index, c_train_dist = full_index_dist(len(chromosomes),chrom_len)
        c_valid_index, c_valid_dist = full_index_dist(len(chromosomes),chrom_len)
    if split == 'BOTH':
        (s_train_index, s_valid_index,
         s_train_dist, s_valid_dist) = sample_index_dist(len(samples),
                                                        valid_frac)
        (c_train_index, c_valid_index,
         c_train_dist, c_valid_dist) = sample_index_dist(len(chromosomes),
                                                        valid_frac, chrom_len)
    return (s_train_index, s_valid_index, c_train_index, c_valid_index,
            s_train_dist, s_valid_dist, c_train_dist, c_valid_dist)


def get_path(cell: str, feature: str, 
             pred_res: int, pth: str = '',
             user_pths: bool = False) -> pathlib.Path:
    """Looks up path for given feature of a cell in the data paths"""
    # Get full length cell name from synonym
    #get key from value
    if not user_pths or cell == 'avg':
        if cell != 'avg':
            # Load feature path
            if feature in ["A", "C", "G", "T"]:
                return DNA_DATA[feature]
            elif feature == "h3k27ac":
                return H3K27AC_DATA[cell+'_'+str(pred_res)]
            elif feature == "h3k4me1":
                return H3K4ME1_DATA[cell+'_'+str(pred_res)]
            elif feature == "h3k4me3":
                return H3K4ME3_DATA[cell+'_'+str(pred_res)]
            elif feature == "h3k9me3":
                return H3K9ME3_DATA[cell+'_'+str(pred_res)]
            elif feature == "h3k27me3":
                return H3K27ME3_DATA[cell+'_'+str(pred_res)]
            elif feature == "h3k36me3":
                return H3K36ME3_DATA[cell+'_'+str(pred_res)]
            elif feature == "chrom_access_embed":
                return ATAC_DATA[cell+'_'+str(pred_res)]
            elif feature == "atac":
                return ATAC_DATA[cell+'_'+str(pred_res)]
            else:
                raise ValueError(
                    f"Feature {feature} not allowed. Allowed features are {ALLOWED_FEATURES}"
                )
        else: #average wanted
            # Load feature path
            if feature in ["A", "C", "G", "T"]:
                return DNA_DATA[feature]
            elif feature == "chrom_access_embed":
                return AVG_DATA_PATH['atac']
            else: #main case, get avg
                return AVG_DATA_PATH[feature]
    else:#user passed the path to the bigwig file with the cell        
        #this option is only to load chromatin accessibility and DNA
        # Load feature path
        if feature in ["A", "C", "G", "T"]:
            return DNA_DATA[feature]
        elif feature == "chrom_access_embed":
            return pth
        elif feature == "atac":
            return pth
        else:
            raise ValueError(
                f"Feature {feature} not allowed when passing file paths, use `./EnformerCelltyping/constants.py` instead."
            )
            
def load_bigwig(path: Union[os.PathLike, str], decode: bool = False):
    """Loads bigwig from a pathlike object"""
    path = str(path)
    if decode:
        try:
            path = path.decode("utf-8")
        except:
            pass
    return pyBigWig.open(path)

def create_buffer(window_size: int, pred_res: int, pred_prop: float = 0.3):
    """
    Define the size (bp) of the buffer. 
    The buffer is the bp's of the input for which an output isn't predicted as they
    fall on the edge of the input window. These input bp's instead just inform the
    predictions for the other bp's.
    """
    buffer_bp = (window_size*(1-pred_prop))/2
    #buffer length needs to also be divisible by pred res so get closest value:
    buffer_bp = int(round(buffer_bp/pred_res)*pred_res)
    target_bp = window_size-(2*(buffer_bp))
    target_length = int(target_bp/pred_res)
    #return number of base-pairs for the buffer and 
    #the number of positions to predict across (bp's/predicted resolution)
    #and the number of bp's to predict across
    return buffer_bp, target_length, target_bp

def one_hot_encode_dna(seq: str):
    """
    One-hot encode DNA sequence.
    """
    dna_bases = np.array(['A','C','G','T'])
    seq = seq.upper()
    all_one_hot = []
    for seq_i in seq:
        assert seq_i in dna_bases, "Sequence must be one of 'A','C','G','T'"
        one_hot = np.zeros((1, len(dna_bases)))
        one_hot[0,np.where(seq_i==dna_bases)]=1
        all_one_hot.append(one_hot)
    all_one_hot = np.concatenate(all_one_hot)
    #only return 2d array if need to
    if(all_one_hot.shape[0]==1):
        return all_one_hot[0]
    return all_one_hot

def rev_comp_dna(seq,one_hot=False,dtype=np.float32):
    """
    Get the reverse complement of an input DNA sequence 
    whether it is bases or one-hot encoded DNA seq.
    """
    if not one_hot:
        seq = seq.upper()
        bases_hash = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N"
        }
        #reverse order and get complement
        rev_comp = "".join([bases_hash[s] for s in reversed(seq)])
    
    else:
        #input is a numpy array
        convert_back_1d = False
        if seq.ndim==1:
            seq = seq.reshape([1,seq.shape[0]])
            convert_back_1d = True
        rev_comp = np.flip(seq,axis=0)
        rev_comp_old = rev_comp.copy()
        rev_comp[(rev_comp_old[:, 0] == 1)] = [0,0,0,1]
        rev_comp[(rev_comp_old[:, 3] == 1)] = [1,0,0,0]
        rev_comp[(rev_comp_old[:, 1] == 1)] = [0,0,1,0]
        rev_comp[(rev_comp_old[:, 2] == 1)] = [0,1,0,0]
        if convert_back_1d:
            rev_comp = rev_comp[0]
    
    return rev_comp

def random_exclude(exclude: set,chromosome_len: int, window_size: int,
                   init_bigwigs: dict,#pybigwig object
                   labels: list,selected_chromosome: chr,
                   selected_cell: chr = False, check_peak: bool = False,
                   peak_prob: float = 0.5,peak_cutoff: float = 2.0, 
                   peak_centre: float = 1.0,
                   dna_wind: int=0,pred_res: int=25,pred_prop: float=0.3,
                   breaker: int=0, max_breaker: int=100,
                   arcsin_trans: bool = True):
    """Random sampling excluding specific values"""
    randInt = int(
            np.random.randint(low=0, high=chromosome_len - window_size, size=1)
        )
    # if any of the vision of the model in a blacklist region, ignore it
    # so large range - max range of the model, even though nodes won't cover it all
    in_blacklist = any(max(i.start,
                           randInt-window_size) < min(i.stop,randInt+window_size) for i in exclude)
    #check if we should only consider regions with a peak in the data
    if(check_peak):
        #get buffer values
        #work out buffer where preds won't be made
        buffer_bp,target_length,target_bp = create_buffer(window_size=dna_wind,
                                                          pred_res=pred_res,
                                                          pred_prop= pred_prop)
        #ensure window size divis by pred res
        window_size_calc = (window_size//pred_res)*pred_res
        #and centre it
        diff = window_size - window_size_calc
        if(diff>1):
            buff_res = (window_size - window_size_calc)//2
        else:
            buff_res = 0
        #now check how many peaks
        peak_count=[]
        for label in labels:
            if arcsin_trans:
                val = np.arcsinh(np.mean(
                                np.nan_to_num(
                                    init_bigwigs[selected_cell][label].values(
                                        selected_chromosome,
                                        randInt+buffer_bp+buff_res,
                                        randInt+window_size_calc-buffer_bp+buff_res,
                                        numpy=True
                                    )
                                ).reshape(-1, pred_res),#averaging at desired pred_res 
                                axis=1))
            else:
                val = np.mean(
                                np.nan_to_num(
                                    init_bigwigs[selected_cell][label].values(
                                        selected_chromosome,
                                        randInt+buffer_bp+buff_res,
                                        randInt+window_size_calc-buffer_bp+buff_res,
                                        numpy=True
                                    )
                                ).reshape(-1, pred_res),#averaging at desired pred_res 
                                axis=1)
            #centre based on prop wanted
            centre_length = int(target_length*peak_centre)
            rmv_edge = (target_length-centre_length)//2
            #remember this is arcsinh trans peak value
            peak_count.append(np.max(val[rmv_edge:target_length-rmv_edge])>peak_cutoff)
        #check prop with a peak 
        has_peak = sum(peak_count)/len(peak_count)>peak_prob
        if(has_peak==False): #use breaker in case just no values fit critera ever
            breaker+=1
        if(breaker>max_breaker): 
            print(f"Peak not found after {max_breaker} iterations." )
            has_peak=True    
    else:
        has_peak=True
    return random_exclude(exclude,chromosome_len, window_size,init_bigwigs,
                          labels,selected_chromosome,selected_cell,
                          check_peak,peak_prob,peak_cutoff,peak_centre,
                          dna_wind,pred_res,pred_prop, 
                          breaker=breaker,
                          max_breaker=max_breaker,
                          arcsin_trans=arcsin_trans) if (in_blacklist==True or has_peak==False) else randInt 

def load_y(data: dict,#pybigwig object
           target_length:int,labels: list,cells: list,
           selected_chromosome: chr,selected_cell: chr,
           window_start: int,buffer_bp: int,
           window_size: int,pred_res: int, arcsin_trans: bool,
           debug: bool,rtn_y_avg: bool = False):
    """Function to load y labels from bigwigs"""
    #ensure window size divis by pred res
    window_size_calc = (window_size//pred_res)*pred_res
    #and centre it
    diff = window_size - window_size_calc
    if(diff>1):
        buff_res = (window_size - window_size_calc)//2
    else:
        buff_res = 0   
    if rtn_y_avg:
        cells = ['avg']
        selected_cell='avg'
    # Output labels only for selected cells
    all_y = np.zeros(shape=(target_length, len(labels)))
    for i, label in enumerate(labels):
        #data at pred_res bp lvl already but loaded in at 1bp lvl
        #need to avg back up!
        #also data is arcsinh transformed to deal help with different seq depths
        if arcsin_trans:
            all_y[:, i] = np.arcsinh(np.mean( 
                np.nan_to_num(
                    data[selected_cell][label].values(
                        selected_chromosome,
                        window_start+buffer_bp+buff_res,
                        window_start + window_size_calc - buffer_bp+buff_res,
                        numpy=True
                    )
                ).reshape(-1, pred_res),#averaging at desired pred_res  
                axis=1))
        else:
            all_y[:, i] = np.mean( 
                np.nan_to_num(
                    data[selected_cell][label].values(
                        selected_chromosome,
                        window_start+buffer_bp+buff_res,
                        window_start + window_size_calc - buffer_bp+buff_res,
                        numpy=True
                    )
                ).reshape(-1, pred_res),#averaging at desired pred_res  
                axis=1)
    return all_y

def load_chrom_access_prom(data: dict,
                           selected_cell: str,
                           feature: str,
                           n_genomic_positions: int = 1216*3_000,#1216 PanglaoDB marker genes 
                           up_dwn_stream_bp: int = 3_000 #num bp to consider at each prot coding gene
                           ):
    """
    Get chromatin accessibility for bps upstream from
    promoters of protein coding genes. This should give
    a better representation of cell types than using
    chromatin accessibility around DNA.
    
    gene_tss obtained using following in R:
    
    library(GenomicFeatures)
    library(GenomicRanges)
    library(data.table)
    library(org.Hs.eg.db)
    refgene <- TxDb.Hsapiens.UCSC.hg19.knownGene::TxDb.Hsapiens.UCSC.hg19.knownGene
    transcripts <- transcripts(refgene, columns=c("gene_id"))
    tss <- resize(transcripts, width=1, fix='start')
    tss <- tss[seqnames(tss) %in% paste0("chr",1:22)]
    tss <- as.data.table(tss)
    tss[,gene_id:=unlist(lapply(tss$gene_id,function(x) x[1]))]
    tss <- unique(tss)
    tss <- tss[!is.na(gene_id)]
    tss[,c("end","width"):=NULL]
    #add gene symbol
    entrez<-mapIds(org.Hs.eg.db, keys=tss$gene_id, column="SYMBOL", 
                   keytype="ENTREZID",#"SYMBOL", 
                   multiVals="first")
    tss[,gene_symbol:=entrez]
    tss <- tss[!is.na(gene_symbol)]
    #make sure 1 entry per gene
    tss <- unique(tss, by = "gene_id")

    """
    gene_tss = pd.read_csv(METADATA_PATH / "tss_hg19.csv.gz", delimiter=',')
    #remove everything except chromosomes 1-22
    gene_tss = gene_tss[gene_tss['seqnames'].isin(CHROMOSOMES)].sort_values(['seqnames','start'])
    #gene_tss = gene_tss[gene_tss['seqnames']==chrom].sort_values(['seqnames','start'])
    # Exclude blacklist regions
    blacklist_regions = load_bigwig(BLACKLIST_PATH)
    keep = []
    #if blacklist within 5k remove
    for index, row in gene_tss.iterrows():
        if(blacklist_regions.entries(row['seqnames'],row['start']-5000,
                                     row['start']) is None):
            keep.append(index)
    gene_tss = gene_tss.loc[keep]
    #number of bp upstream dependent on n_genomic_positions wanted and number of genes
    #assert n_genomic_positions%up_dwn_stream_bp==0,"n_genomic_positions has to be divisible by up_dwn_stream_bp"
    assert up_dwn_stream_bp%250==0,"up_dwn_stream_bp has to be divisible by 250"
    assert n_genomic_positions>up_dwn_stream_bp,"n_genomic_positions should be > up_dwn_stream_bp"
    #using PanglaoDB - https://panglaodb.se/markers.html?cell_type=%27all_cells%27#google_vignette
    #to prioritise marker genes for cell typing
    #filtering to human marker genes and check specificity
    marker_genes = pd.read_csv(METADATA_PATH / "PanglaoDB_markers_27_Mar_2020.tsv.gz", 
                               delimiter='\t')
    marker_genes = marker_genes[marker_genes['species'].isin(['Mm Hs','Hs'])]
    #filter to just those genes in gene_tss
    marker_genes = marker_genes[marker_genes['official gene symbol'].isin(gene_tss['gene_symbol'])]
    #make sure 1 row per gene symbol
    marker_genes = marker_genes.drop_duplicates(subset=['official gene symbol'])
    gene_tss_mg = gene_tss.merge(marker_genes,how='left',left_on=['gene_symbol'],
                              right_on=['official gene symbol'])
    gene_tss_mg['sensitivity_human'] = gene_tss_mg['sensitivity_human'].fillna(0)
    #filter to where there is any sensitivity
    out = gene_tss_mg[gene_tss_mg['sensitivity_human']>0].sort_values(['seqnames','start'])
    num_bp = out.shape[0]*up_dwn_stream_bp
    assert n_genomic_positions == num_bp,"Calculated number of base-pairs not equal to inputted n_genomic_positions"
    #1216 marker genes each with 3_000 bp
    chrom_access_list = []
    for index, row in out.iterrows():
        chrom_access_list.append(data[selected_cell][feature].values(
            row['seqnames'], row['start']-(up_dwn_stream_bp//2), 
            row['start']+(up_dwn_stream_bp//2),numpy=True))
    
    return(chrom_access_list)  
    

def initiate_bigwigs(
    cells: Sequence[str],
    cell_probs: Sequence[float],
    chromosomes: Sequence[str],
    chromosome_probs: Sequence[float],
    features: Sequence[str] = ["A", "C", "G", "T","chrom_access_embed"],
    labels: Sequence[str] = ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3'],
    pred_res: int = 128,
    load_avg=True,
    user_pths=False,
):
    """
    Initiate the connection to the bigwigs
    
    Arguments:
        cells:
            List of all cells to be used. Just pass the cell of interest in a list if 
            loading a known position. Also pass a dictionary with file path if just 
            loading chromatin acessibility and DNA to pass to Enformer Celltyping for
            predictions, also set `user_pths=True`
        cell_probs:
            List of probabilitie of randomly choosing cells, should be same 
            length as cells. Just pass [1] if using a known position for 
            one cell.
        chromosomes:
            List of chromosomes to be used if randomly sampling.
        chromosome_probs:
            List of probabilities of randomly choosing chromosomes, should be 
            same length as chromosomes.
        features:
            List of input features (X).
        labels:
            List of output labels i.e. histone marks to predict (Y).
        pred_res:
            Predicted resolution of the output (Y) data.
        load_avg:
            Whether to laod the average from the training cell types.
        user_pths:
            Whether to get the paths for the data files from the constants file 
            (`./EnformerCelltyping/constants.py`) - False or from the user on 
            input - True.
    """
    # Input verification
    if not all(np.isin(features, ALLOWED_FEATURES)):
        raise ValueError(
            "Features contain values which are not allowed. "
            f"Allowed features are {ALLOWED_FEATURES}."
        )
    if not all(np.isin(labels, ALLOWED_FEATURES)):
        raise ValueError(
            "Labels contain values which are not allowed. "
            f"Allowed labels are {ALLOWED_FEATURES}."
        )
    # make sure chrom access embed last track
    # location necessary for operations
    if (np.isin(['chrom_access_embed'],features)):
        features.append(features.pop(features.index('chrom_access_embed')))      
    
    if load_avg:
        if not user_pths: #if list, i.e. getting paths from constants.py
            cells = np.append(cells,'avg')
        else: #user passed paths so is a dict    
            cells['avg'] = 'a/fake/pth' #path for avg will be taken from constants.py still
        cell_probs = np.append(cell_probs,1)
    
    assert len(cells) == len(cell_probs), "Must provide probabilities for all cells"
    assert len(chromosomes) == len(
        chromosome_probs
    ), "Must provide probabilities for all chromosomes"
    assert len(features) > 0, "Must provide at least one feature"
    
    # Load file handles for all cells and features
    if not user_pths:
        data = {
            cell: {
                feature:load_bigwig(get_path(cell, feature, pred_res))
                for feature in features + labels
            }
            for cell in cells
        }
    else: #user gave dict with paths too
        data = {
            cell: {
                feature:load_bigwig(get_path(cell, feature, pred_res,
                                             pth, user_pths))
                for feature in features + labels
            }
            for cell,pth in cells.items()
        }
    return data
    
    

def generate_data(
    cells: Sequence[str],
    data: dict,    
    cell_probs: Sequence[float] = np.repeat(1, 1),
    chromosomes: Sequence[str] = CHROMOSOMES,
    chromosome_probs: Sequence[float] = np.repeat(1, len(CHROMOSOMES)),
    features: Sequence[str] = ["A", "C", "G", "T","chrom_access_embed"],
    labels: Sequence[str] = ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3'],
    num_samps: int = 1,
    window_size: int = 196_608,
    pred_res: int = 128,
    arcsin_trans: bool = True,
    debug: bool = False,
    reverse_complement=False,
    rand_seq_shift=False,
    rand_seq_shift_amt=None,
    rtn_rand_seq_shift_amt=False,
    pred_prop: float =(128*896)/196_608,
    rand_pos: bool =True,
    chro: int =0,
    pos: int =0,
    cell: str='None',
    n_genomic_positions=1562*128, #local Chromatin Accessibility size
    return_y: bool = True,
    check_peak: bool = False,
    peak_prob: float = 0.5,
    peak_cutoff: float = 2.0,
    peak_centre: float = 1.0,
    max_rand_perms: int = 100,
    data_trans = False, #should a data transformation be applied to the DNA? Pass model (Enformer) to do it
    rtn_y_avg = False, #returns the training cells' avg signal as well as the true signal for the position
    up_dwn_stream_bp: int = 3_000, #num bp's to use around TSS of prot coding genes
    snp_pos: int = None,
    snp_base: chr = None
):
    """
    Generic data loader to load data from bigwig files on the fly.
    
    Arguments:
        cells:
            List of all cells to be used for randomly sampling a cell-pos.
            Just pass the cell of interest in a list if loading a known 
            position.
        cell_probs:
            List of probabilitie of randomly choosing cells, should be same 
            length as cells. Just pass [1] if using a known position for 
            one cell.
        chromosomes:
            List of chromosomes to be used if randomly sampling.
        chromosome_probs:
            List of probabilities of randomly choosing chromosomes, should be 
            same length as chromosomes.
        features:
            List of input features (X).
        labels:
            List of output labels i.e. histone marks to predict (Y).
        data:
            Dictionary of initialised connections to all necessary bigwig files.
            This is generated with initiate_bigwigs.
        num_samps:
            Number of samples to generate.
        window_size:
            Window size of DNA input. This is the same as Enformer by default.
        pred_res:
            Predicted resolution of the output (Y) data.
        arcsin_trans:
            Boolean, whether Y should be arcsine tranformed to help account for
            differences in sequencing depths of studies for Y values.
        debug:
            Boolean, give informative print statements throughout function. Useful
            for debugging.
        reverse_complement:
            Boolean, should the reverse compliment of the X and Y values also be returned with
            the data?
        rand_seq_shift:
            Boolean, should a randomly shifted version of the X and Y be returned as well as the 
            original data? Not ethe random shift amount will be between 1-3 bp and the Y and 
            chromosome accedssibility data will also be shifted along with the DNA data.
        rand_seq_shift_amt:
            Pass random shift amount to use if you don't want it to be generated - used to replicate
            other runs.
        rtn_rand_seq_shift_amt:
            Whether to return the rand_seq_shift_amt value form the data loader.
        pred_prop: 
            The proportion of the input DNA window that is predicted by the model. These types of models
            don't predict in the edge of the input window since it won't have enough information upstream 
            and downstream to make an accurate prediction. This proportion is equal to that of Enformer's 
            by default: (128*896)/196_608.
        rand_pos:
            Boolean, whether to randomly choose a position based on numerous filters or to go with a 
            pre-designated position.
        chro:
            Chromosome if going with a pre-designated position.
        pos:
            Position if going with a pre-designated position. This position will form the start of DNA 
            input.
        cell:
            Cell if going with a pre-designated position. This position will form the start of DNA 
            input.
        n_genomic_positions:
            Number of base-pairs to use for the local Chromatin Accessibility information. This will be 
            wrapped around the DNA window so that the DNA window is centered in it.
        return_y:
            Boolean, whether to return Y or just input (X) data - this will speed up function if only X is 
            needed.
        check_peak:
            Boolean, if randomly choosing positions, should certain aspects of the peaks in regions be 
            checked?
        peak_prob:
            Proportion of prediction tracks (Y) with at least 1 peak in the region.
        peak_cutoff: 
            Cut-off to identify peaks, **note** this is the arcsinh transformed peak value.
        peak_centre:
            Does where the peak is situated matter? If you pass a value <1, indicates proportion of the 
            centre of the output (Y) to look in when checking for peaks.
        max_rand_perms: 
            Maximum number of attempts at randomly sampling positions to find the chosen, specific features 
            in the random region.
        data_trans:
            Should the DNA input (X) data be transformed by passing through the pre-trained and chopped 
            Enformer model? Pass this model to do it. The benefit would be the training/predicting requires
            less RAM than doing so in the model itself.
        rtn_y_avg:
            Boolean, whether to return the chosen postions' avg signal from all training cells as well as the 
            true signal for the cell in question. This is used for training the two channel model of Enformer 
            Celltyping but not for predicting with it.
        up_dwn_stream_bp:
            Number of base-pair's to use around TSS of protein coding genes for global chromatin accessibility?
        snp_pos: 
            If imputing a SNP in the DNA data, what position to impute it, relative to the dna start 
            position?
        snp_base:
            If imputing a SNP in the DNA data, what nucleotide do you want to impute?
    """
    # Input verification
    if not all(np.isin(features, ALLOWED_FEATURES)):
        raise ValueError(
            "Features contain values which are not allowed. "
            f"Allowed features are {ALLOWED_FEATURES}."
        )
    if not all(np.isin(labels, ALLOWED_FEATURES)):
        raise ValueError(
            "Lables contain values which are not allowed. "
            f"Allowed labels are {ALLOWED_FEATURES}."
        )
    # make sure chrom access embed last track
    # location necessary for operations
    if (np.isin(['chrom_access_embed'],features)):
        features.append(features.pop(features.index('chrom_access_embed')))       
    assert len(features) > 0, "Must provide at least one feature"
    
    if len(cells)>1 and len(cell_probs)==1:
        np.repeat(1, len(cells))
    
    # At each generator iteration:
    while True:
        samps_X =[]
        samps_embed =[]
        samps_embed_gbl =[]
        samps_y =[]
        samps_avg_y =[]
        for samp in range(num_samps):
            #if adding random perm, load once slightly large then subset
            if rand_seq_shift:
                #up to 3 nucleotides either side
                #allowed_values = list(range(-3, 3+1)) #up to not incl second number
                #allowed_values.remove(0)
                #only need positive values either way since this is dist from act to rand shift
                if rand_seq_shift_amt is None:
                    rand_shift_amt = random.choice(range(1, 3+1))#up to not incl second number
                else:
                    rand_shift_amt = rand_seq_shift_amt
            else:
                rand_shift_amt = 0

            if debug:
                print(f"Random shift: {rand_shift_amt} nucleotides")    

            #randomly select the chromosome and position
            if rand_pos:
                # Select cell
                selected_cell = np.random.choice(cells, replace=True, p=cell_probs)
                if debug:
                    print(f"Selected cell: {selected_cell}")

                # Select chromosome
                selected_chromosome = np.random.choice(
                    chromosomes, replace=True, p=chromosome_probs
                )
                chromosome_len = CHROMOSOME_DATA[selected_chromosome]
                if debug:
                    print(f"Selected chromosome: {selected_chromosome}")

                # Exclude blacklist regions
                blacklist_regions = load_bigwig(BLACKLIST_PATH)
                blacklist_chr = blacklist_regions.entries(selected_chromosome,0,chromosome_len)
                # Need to get all blacklist regions in the chormosome of interest
                blacklist_chr_positions = [range(i[0],i[1]) for i in blacklist_chr]
                # Select window to read - exclude blacklist positions
                # Also only choose regions with a peak for training if specified
                # if embedding length bigger than window size, that's the window for sampling
                # if using promoter regions for chrom access, don't include n_genomic_positions in blacklist
                if(window_size<n_genomic_positions): 
                    window_start = random_exclude(exclude=blacklist_chr_positions, 
                                                  chromosome_len=chromosome_len, 
                                                  window_size=n_genomic_positions+rand_shift_amt,
                                                  check_peak=check_peak,peak_prob=peak_prob,
                                                  peak_cutoff=peak_cutoff,peak_centre=peak_centre,
                                                  max_breaker=max_rand_perms,
                                                  dna_wind = window_size,init_bigwigs = data,
                                                  pred_res=pred_res,pred_prop=pred_prop,
                                                  selected_cell=selected_cell, 
                                                  selected_chromosome=selected_chromosome,
                                                  labels = labels,
                                                  arcsin_trans=arcsin_trans)
                   
                else:    
                    window_start = random_exclude(exclude=blacklist_chr_positions, 
                                                  chromosome_len=chromosome_len, 
                                                  window_size=window_size+rand_shift_amt,
                                                  check_peak=check_peak,peak_prob=peak_prob,
                                                  peak_cutoff=peak_cutoff,peak_centre=peak_centre,
                                                  max_breaker=max_rand_perms,
                                                  dna_wind = window_size,init_bigwigs = data,
                                                  pred_res=pred_res,pred_prop=pred_prop,
                                                  selected_cell=selected_cell, 
                                                  selected_chromosome=selected_chromosome,
                                                  labels = labels,
                                                  arcsin_trans=arcsin_trans)
            else:
                selected_chromosome=chro
                window_start=pos
                #for predicting in 1 new cell type
                selected_cell=cells

            if debug:
                print(f"Selected window: {window_start}")
                print(f"Selected chromosome: {selected_chromosome}")
            
            # Load the data
            # X data (features)
            #if none of A,T,C,G passed, don't run dna functionality
            dna_feat = True 
            for dna_i in ['A','T','C','G']:
                if dna_i not in features:
                    dna_feat=False
                    
            #passes dna and data to represent cell types
            all_X = np.zeros(shape=(window_size + rand_shift_amt, len(features)-1))
            if len(features)==4: #just base pairs
                all_X = np.zeros(shape=(window_size + rand_shift_amt, len(features)))                
            #also data is arcsinh transformed to deal help with different seq depths
            if window_size>n_genomic_positions or rand_pos==False: 
                #rand_pos=False since rand pos positioning based on dist ## 
                #note check y pred from this change, marked below - dna_start shifted
                dna_start = window_start
            else:
                #window_start is for n_genomic_pos, we want to centre the DNA on that
                dna_start = window_start+(n_genomic_positions//2)-window_size//2    
            for i, feature in enumerate(features):
                #load full window including rand_shift_amt so don't have to load twice
                if(feature!='chrom_access_embed'):
                    all_X[:, i] = data[selected_cell][feature].values(
                        selected_chromosome, dna_start, 
                        dna_start + window_size + rand_shift_amt,
                        numpy=True
                    )
                #if using cell type embedding
                elif(feature=='chrom_access_embed'):
                    # Make tracks for diff resolutions
                    # also data is arcsinh transformed to deal help with different seq depths
                    #load global chrom access data then augment
                    #Upstream prom TSS for all protein coding genes
                    chrom_access_raw = load_chrom_access_prom(
                        data = data,
                        selected_cell = selected_cell,
                        feature = feature,
                        #n_genomic_positions =n_genomic_positions, #n_genomic_positions defined by num marker genes
                        up_dwn_stream_bp = up_dwn_stream_bp, # num bp's to use at TSS for embedding
                    )
                    chrom_access_global = chrom_access_raw    
                    chrom_acc_strt = window_start
                    if not rand_pos: #same logic as above make sure CA around DNA
                        chrom_acc_strt = window_start + window_size//2 -(n_genomic_positions//2)
                    chrom_acc_end = chrom_acc_strt + n_genomic_positions
                    if window_size>n_genomic_positions:
                        #work out buffer
                        buff_bp_CA = (window_size - n_genomic_positions)//2
                        #ensure window size divis by pred res
                        window_size_calc = (window_size//pred_res)*pred_res
                        #and centre it
                        diff = window_size - window_size_calc
                        if(diff>1):
                            buff_res = (window_size - window_size_calc)//2
                        else:
                            buff_res = 0
                        chrom_acc_strt = window_start+buff_bp_CA+buff_res
                        chrom_acc_end = window_start + window_size_calc - buff_bp_CA+buff_res
                    #around random dna seq
                    chrom_access_raw = data[selected_cell][feature].values(
                        selected_chromosome, chrom_acc_strt, 
                        chrom_acc_end,numpy=True)
                    #load avg - want input to be delta = act-avg
                    chrom_access_avg = data['avg'][feature].values(
                        selected_chromosome, chrom_acc_strt, 
                        chrom_acc_end,numpy=True)
                    #load data and get at correct pred res
                    cell_embed = np.arcsinh(np.float32(np.mean(#enformer works with float32 not 64
                        np.nan_to_num(chrom_access_raw).reshape(-1, pred_res),#bp res
                        axis=1)))
                    #now reshape global to get 250bp avg
                    cell_embed_gbl = np.arcsinh(np.float32(np.mean(#enformer works with float32 not 64
                        np.nan_to_num(chrom_access_global).reshape(-1, 250),#250 bp res should be enough
                        axis=1)))
                    #actual minus avg
                    cell_embed = cell_embed - np.arcsinh(np.float32(np.mean(#enformer works with float32 not 64
                        np.nan_to_num(chrom_access_avg).reshape(-1, pred_res),#bp res
                        axis=1)))
            #check if we need to impute a snp
            if snp_pos is not None:
                #get one-hot encoded version
                one_hot_alt = one_hot_encode_dna(snp_base)
                all_X[snp_pos,:] = one_hot_alt
            #if random shifting of pos happening make sep inputs
            #create list of the different datasets
            X = []
            #create list for embedding if using
            if (np.isin(['chrom_access_embed'],features)):        
                embed_X = [] 
                embed_X_gbl = []
            if rand_seq_shift:
                #Now subset this full load as necessary
                #append org
                X.append(all_X[:-rand_shift_amt])
                #append random shift
                X.append(all_X[rand_shift_amt:])
                #embedding - do it. twice since the same for rand seq diff
                embed_X_gbl.append(cell_embed_gbl)
                embed_X_gbl.append(cell_embed_gbl)
                #around random dna seq
                chrom_access_raw_shift = data[selected_cell][feature].values(
                    selected_chromosome, chrom_acc_strt+rand_shift_amt, 
                    chrom_acc_end+rand_shift_amt,
                    numpy=True)
                chrom_access_avg_shift = data['avg'][feature].values(
                    selected_chromosome, chrom_acc_strt+rand_shift_amt, 
                    chrom_acc_end+rand_shift_amt,
                    numpy=True)
                #always arcsinh trans chrom access input
                cell_embed_shift = np.arcsinh(np.float32(np.mean(
                np.nan_to_num(chrom_access_raw_shift).reshape(-1, pred_res),
                axis=1)))
                #actual minus avg
                cell_embed_shift = cell_embed_shift - np.arcsinh(np.float32(np.mean(
                    np.nan_to_num(chrom_access_avg_shift).reshape(-1, pred_res),#bp res
                    axis=1)))
                embed_X.append(cell_embed)
                embed_X.append(cell_embed_shift)    
            else:
                #only passing org input without rand shift input
                X.append(all_X)
                #embedding - only pass once
                if (np.isin(['chrom_access_embed'],features)):
                    embed_X.append(cell_embed)
                    embed_X_gbl.append(cell_embed_gbl)
            # y data (labels)
            #work out buffer where preds won't be made
            buffer_bp,target_length,target_bp = create_buffer(window_size=window_size,
                                                              pred_res=pred_res,
                                                              pred_prop= pred_prop)          
            #window_start is for n_genomic_pos, we want to centre the DNA on that
            if window_size>n_genomic_positions or rand_pos==False:
                dna_start = window_start
            else:
                dna_start = window_start+(n_genomic_positions//2)-window_size//2    
                
            if return_y:
                all_y = load_y(data=data,
                               target_length=target_length,labels=labels,
                               cells=cells,selected_chromosome=selected_chromosome,
                               selected_cell=selected_cell,
                               window_start=dna_start,
                               buffer_bp=buffer_bp,window_size=window_size,
                               pred_res=pred_res,arcsin_trans=arcsin_trans,
                               debug=debug)
                if rtn_y_avg:
                    all_y_avg = load_y(data=data,
                                       target_length=target_length,labels=labels,
                                       cells=cells,selected_chromosome=selected_chromosome,
                                       selected_cell=selected_cell,window_start=dna_start,
                                       buffer_bp=buffer_bp,window_size=window_size,
                                       pred_res=pred_res,arcsin_trans=arcsin_trans,
                                       debug=debug,rtn_y_avg=True)
                #if random shifting of pos happening make sep inputs
                #can't use same trick as with x as averaging with buffer so just 
                #load random shift separate and match with corresponding entry
                y = []
                y_avg = []
                if rand_seq_shift:
                    all_y2 = load_y(data=data,
                                    target_length=target_length,labels=labels,
                                    cells=cells,selected_chromosome=selected_chromosome,
                                    selected_cell=selected_cell,
                                    window_start=dna_start+rand_shift_amt,#add random shift
                                    buffer_bp=buffer_bp,window_size=window_size,
                                    pred_res=pred_res,arcsin_trans=arcsin_trans,
                                    debug=debug)
                    #append org
                    y.append(all_y)
                    #append random shift
                    y.append(all_y2)
                    
                    if rtn_y_avg:
                        all_y2_avg = load_y(data=data,
                                            target_length=target_length,labels=labels,
                                            cells=cells,selected_chromosome=selected_chromosome,
                                            selected_cell=selected_cell,
                                            window_start=dna_start+rand_shift_amt,#add random shift ##
                                            buffer_bp=buffer_bp,window_size=window_size,
                                            pred_res=pred_res,arcsin_trans=arcsin_trans,
                                            debug=debug,rtn_y_avg=True)
                        #append org
                        y_avg.append(all_y_avg)
                        #append random shift
                        y_avg.append(all_y2_avg)
                else:
                    y.append(all_y)
                    if rtn_y_avg:
                        y_avg.append(all_y_avg)

            # if training on reverse complement, calculate and add it in
            # we will get the reverse complement of the actual and randomly shifted input 
            if reverse_complement:
                #in case random perm added:
                org_len_X = len(X)
                if dna_feat:
                    for x_i in range(org_len_X):
                        X.append(rev_comp_dna(X[x_i],
                                              one_hot=True))
                        #X.append(X[x_i][::-1])    
                #same for y
                if return_y:
                    org_len_y = len(y)
                    for y_i in range(org_len_y):
                        y.append(y[y_i][::-1])
                    if rtn_y_avg:
                        for y_i in range(org_len_y):
                            y_avg.append(y_avg[y_i][::-1])
                #For embedding - don't reverse if using prom marker genes
                for x_i in range(org_len_X):
                    #local chrom access so do rev
                    embed_X.append(embed_X[x_i][::-1])
                    #global
                    embed_X_gbl.append(embed_X_gbl[x_i][::-1])            
            #return
            #enformer works with float32 not 64
            samps_X.append(tf.stack([np.float32(tf.convert_to_tensor(x_i.copy())) for x_i in X]))
            samps_embed.append(tf.stack([tf.convert_to_tensor(x_i.copy()) for x_i in embed_X]))
            samps_embed_gbl.append(tf.stack([tf.convert_to_tensor(x_i.copy()) for x_i in embed_X_gbl]))
            if return_y:
                samps_y.append(tf.stack([np.float32(tf.convert_to_tensor(y_i.copy())) for y_i in y]))
                if rtn_y_avg:
                    samps_avg_y.append(tf.stack([np.float32(tf.convert_to_tensor(y_i.copy())) for y_i in y_avg]))
        
        #apply Enformer transformation
        if data_trans != False and dna_feat:
            trans_samps_X = []
            for dna in samps_X:
                trans_samps_X.append(data_trans.predict_on_batch(dna))
            samps_X = trans_samps_X    
        # finally return with multiple samples
        if (not(np.isin(['chrom_access_embed'],features))):
            if return_y:
                yield(tf.concat(samps_X,axis=0),tf.concat(samps_y,axis=0))
            else:
                if rtn_rand_seq_shift_amt:
                    yield(tf.concat(samps_X,axis=0),
                          rand_shift_amt)
                else:        
                    yield(tf.concat(samps_X,axis=0))
        else:
            if return_y:
                if rtn_y_avg:
                    yield({"dna":tf.concat(samps_X,axis=0),
                           "chrom_access_lcl":tf.concat(samps_embed,axis=0),
                           "chrom_access_gbl":tf.concat(samps_embed_gbl,axis=0)
                          },
                          {"act":tf.concat(samps_y,axis=0),
                           "avg":tf.concat(samps_avg_y,axis=0)})
                else:
                    yield({"dna":tf.concat(samps_X,axis=0),
                           "chrom_access_lcl":tf.concat(samps_embed,axis=0),
                           "chrom_access_gbl":tf.concat(samps_embed_gbl,axis=0)
                          },
                          tf.concat(samps_y,axis=0))
            else:
                if rtn_rand_seq_shift_amt:
                    yield({"dna":tf.concat(samps_X,axis=0),
                       "chrom_access_lcl":tf.concat(samps_embed,axis=0),
                       "chrom_access_gbl":tf.concat(samps_embed_gbl,axis=0)    
                      },rand_shift_amt)
                else:
                    yield({"dna":tf.concat(samps_X,axis=0),
                       "chrom_access_lcl":tf.concat(samps_embed,axis=0),
                       "chrom_access_gbl":tf.concat(samps_embed_gbl,axis=0)
                      })
                    


class generate_sample:
    
    """
    class object for data generator for specific cell, 
    chromosome and positions.
    
    Used as a wrapper for generate_data
    
    Arguments:
        cells:
            Dictionary containing the cell name and file path to the chormatin
            acessibility bigWig. This should only be used when `return_y=FALSE`. 
            **OR** A List of all cells to be used. Just pass the cell of interest 
            in a list if loading a known position.
        chromosomes:
            List of chromosomes to be used if randomly sampling.
        features:
            List of input features (X).
        labels:
            List of output labels i.e. histone marks to predict (Y).
        window_size_dna:
            Window size of DNA input. This is the same as Enformer by default.
        pred_res:
            Predicted resolution of the output (Y) data.
        arcsin_trans:
            Boolean, whether Y should be arcsine tranformed to help account for
            differences in sequencing depths of studies for Y values.
        debug:
            Boolean, give informative print statements throughout function. Useful
            for debugging.
        reverse_complement:
            Boolean, should the reverse compliment of the X and Y values also be returned with
            the data?
        rand_seq_shift:
            Boolean, should a randomly shifted version of the X and Y be returned as well as the 
            original data? Not ethe random shift amount will be between 1-3 bp and the Y and 
            chromosome accedssibility data will also be shifted along with the DNA data.
        rand_seq_shift_amt:
            Pass random shift amount to use if you don't want it to be generated - used to replicate
            other runs.
        rtn_rand_seq_shift_amt:
            Whether to return the rand_seq_shift_amt value form the data loader.
        pred_prop: 
            The proportion of the input DNA window that is predicted by the model. These types of models
            don't predict in the edge of the input window since it won't have enough information upstream 
            and downstream to make an accurate prediction. This proportion is equal to that of Enformer's 
            by default: (128*896)/196_608.
        chro:
            Chromosome if going with a pre-designated position.
        pos:
            Position if going with a pre-designated position. This position will form the start of DNA 
            input.
        cell:
            Cell if going with a pre-designated position. This position will form the start of DNA 
            input.
        window_size_CA:
            Number of base-pairs to use for the local Chromatin Accessibility information. This will be 
            wrapped around the DNA window so that the DNA window is centered in it.
        return_y:
            Boolean, whether to return Y or just input (X) data - this will speed up function if only X is 
            needed.
        data_trans:
            Should the DNA input (X) data be transformed by passing through the pre-trained and chopped 
            Enformer model? Pass this model to do it. The benefit would be the training/predicting requires
            less RAM than doing so in the model itself.
        snp_pos: 
            If imputing a SNP in the DNA data, what position to impute it, relative to the dna start 
            position?
        snp_base:
            If imputing a SNP in the DNA data, what nucleotide do you want to impute?    
    """
    def __init__(self,
                 cells,
                 chromosomes = CHROMOSOMES,
                 features = ["A", "C", "G", "T","chrom_access_embed"],
                 labels = ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3'],
                 window_size_dna = 196_608,
                 pred_res = 128,
                 arcsin_trans = False,
                 reverse_complement = False,
                 rand_seq_shift = False,
                 pred_prop = (128*896)/196_608,
                 window_size_CA = 1562*128,
                 return_y=True,
                 data_trans=False,
                 rtn_rand_seq_shift_amt=False,
                 debug = False
                ):
        self.cells = cells
        if return_y==True and type(cells)==dict:
            return_y=False
            if debug:
                print('Forcing return_y=False, since you can''t use inputted paths in cells with return_y=True')
        user_pths = False
        if type(cells)==dict:
            user_pths = True
        self.user_pths = user_pths
        self.chromosomes = chromosomes
        self.features = features
        self.labels = labels
        if self.user_pths and len(labels)>0:
            self.labels = [] #only load X data
            if debug:
                print('Forcing removal of histone marks, since you can''t use inputted paths in cells with histone marks')
        self.window_size_dna = window_size_dna
        self.pred_res = pred_res
        self.arcsin_trans = arcsin_trans
        self.debug = debug
        self.reverse_complement = reverse_complement
        self.rand_seq_shift = rand_seq_shift
        self.pred_prop = pred_prop
        self.window_size_CA = window_size_CA
        self.return_y = return_y
        self.data_trans = data_trans
        self.rtn_rand_seq_shift_amt = rtn_rand_seq_shift_amt
        #Need delta so load the avg chrom access to minus from act
        self.load_avg=True
        #only initiate connection to bigwigs once, saves on memory
        self.data = initiate_bigwigs(cells=self.cells,
                                     cell_probs=np.repeat(1, len(cells)),
                                     chromosomes=self.chromosomes,
                                     chromosome_probs=np.repeat(1, 
                                                                len(chromosomes)),
                                     features=self.features,
                                     labels=self.labels,
                                     pred_res=self.pred_res,
                                     load_avg = self.load_avg,
                                     user_pths = self.user_pths)
        
    def load(self,
             chro: str,pos: int, cell: str,
             snp_pos: int = None,snp_base: chr = None,
             rand_seq_shift_amt: int = None,
             return_chrom_access: bool = True,
             return_dna: bool = True):
        features_run = self.features.copy()
        if not return_chrom_access:
            features_run.remove('chrom_access_embed')   
        if not return_dna:
            features_run.remove('A')
            features_run.remove('T')
            features_run.remove('C')
            features_run.remove('G')
        dat = next(generate_data(chro=chro, pos = pos, cell = cell,
                            cells = cell,
                            snp_pos=snp_pos,snp_base=snp_base,
                            rand_seq_shift_amt=rand_seq_shift_amt,
                            chromosomes = self.chromosomes,
                            features = features_run,
                            labels = self.labels,
                            data = self.data,
                            window_size = self.window_size_dna,
                            pred_res = self.pred_res,
                            arcsin_trans = self.arcsin_trans,
                            debug = self.debug,
                            reverse_complement = self.reverse_complement,
                            rand_seq_shift = self.rand_seq_shift,
                            pred_prop = self.pred_prop,
                            n_genomic_positions = self.window_size_CA,
                            return_y = self.return_y,
                            data_trans = self.data_trans,
                            rtn_rand_seq_shift_amt = self.rtn_rand_seq_shift_amt,    
                            rand_pos = False,
                            cell_probs=np.repeat(1,len(self.cells)),
                            chromosome_probs = np.repeat(1,
                                                         len(self.chromosomes))
                            ))
        return(dat)


class PreSavedDataGen(tf.keras.utils.Sequence):
    """
    Generate data for training when you have it 
    loaded and saved in npz
    """
    def __init__(self, files, batch_size,
                 shuffle=True):
        self.files = files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.files))
        self.on_epoch_end()
        
    def __getitem__(self, index):
        """
        Get next batch
        """
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # single file
        file_list_temp = [self.files[k] for k in indices]

        # Set of X_train and y_train
        X, y = self.__data_generation(file_list_temp)
        return X, y

    def __data_generation(self, file_list_temp):
        """
        Generates data containing batch_size samples
        """
        # Generate data
        X_dna = []
        X_chrom_access = []
        X_chrom_access_gbl = []
        y = []
        y_avg = []
        epoch_files = file_list_temp
        for ind,ID in enumerate(epoch_files):
            # load
            dat = np.load(ID)
            # ATAC dat stored in same location
            atac_dat = dat
            #load ATAC cell typing data from sep file
            X_chrom_access.append(atac_dat['X_chrom_access'])
            X_chrom_access_gbl.append(atac_dat['X_chrom_access_gbl'])    
            #store - append values
            X_dna.append(dat['X_dna'])
            #y_avg will be predicted by the DNA channel since it won't
            #change between cell types
            y_avg_tmp = dat['y_avg']
            y_avg.append(y_avg_tmp)
            #y is now the difference betweent the act cell type values
            #and the training cells' average
            #this will be predicted by the CT channel
            y.append(dat['y_act']-y_avg_tmp)
        #combine X, y
        X = ({'dna':tf.concat(X_dna,axis=0),
              'chrom_access_lcl':tf.concat(X_chrom_access,axis=0),
              'chrom_access_gbl':tf.concat(X_chrom_access_gbl,axis=0),})
        y = ({'avg':tf.concat(y_avg,axis=0),
          'delta':tf.concat(y,axis=0)})

        return X, y    

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.files) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)           



#2.2 Inspect & using models ------------------------------------------------

from tqdm import tqdm
import copy
import math

def pred_region(model, cell, pos, the_chr, WINDOW,
                features,labels,pred_resolution,
                window_size_dna,window_size_CA,pred_prop,
                data_trans_model: bool = False,
                return_arcsinh: bool = False,
                include_atac: bool = False):
    """
    Predict in a specific region.
    
    Arguments:
    model:
        a compiled tf keras model
    cell: 
        cell to predict in
    pos:
        position to be centred for predicitions
    the_chr:
        chromosome to predict on
    WINDOW:
        total size of prediction window
    features:
        model input features
    labels:
        prediction labels
    pred_resolution:
        model bp resolution
    window_size_dna:
        model input DNA window size
    window_size_CA:
        model input bp for local chromatin accessibility
    pred_prop:
        model prediction proportion of input DNA size
    return_arcsinh:
            Should predictions be returned with arcsinh transformation 
            (general approach is to train Enformer Celltyping on this 
            data but when actually using the data for downstream work, 
            the output should be converted back). By default this
            will be converted back.
    include_atac:
        Should the ATAC chromatin accessibility signal for the region
        also be returned? Default is false.
    """
        
    #work out buffer where preds won't be made
    buffer_bp,target_length,target_bp = create_buffer(window_size=window_size_dna,
                                                      pred_res=pred_resolution,
                                                      pred_prop= pred_prop)
    #initiate data connection
    data_conn = initiate_bigwigs(cells=[cell],
                                 cell_probs=np.repeat(1, 1),
                                 chromosomes=CHROMOSOMES,
                                 chromosome_probs=np.repeat(1, len(CHROMOSOMES)),
                                 features=features,
                                 labels=labels,
                                 pred_res=pred_resolution,
                                 load_avg=True,
                                 training=False)
    #also initiate connection to averages for comparison
    avg_data_conn = {avg: load_bigwig(AVG_DATA_PATH[avg]) for avg in labels}
    #atac connection
    if include_atac:
        atac_conn = initiate_bigwigs(cells=[cell],
                                     cell_probs=np.repeat(1, 1),
                                     chromosomes=CHROMOSOMES,
                                     chromosome_probs=np.repeat(1, len(CHROMOSOMES)),
                                     features=features,
                                     labels=['atac'],
                                     pred_res=pred_resolution,
                                     training=False)
    #Get average, model predictions and true signal
    model_pred = []
    avg_pred = []
    signal = []
    atac = []
    #remember preds made for prop at centre window
    #strt passed to generate data is for start of data fro chrom access
    strt_plot_pos = pos - WINDOW//2
    centre_pred = strt_plot_pos + (target_bp//2)
    strt = centre_pred - window_size_dna//2
    end = pos + WINDOW//2
    X_all = []
    while ((strt+(window_size_dna//2)-(target_bp//2))<end):
        #load average for each cell type
        #ensure window size divis by pred res
        window_size_calc = (window_size_dna//pred_resolution)*pred_resolution
        #and centre it
        diff = window_size_dna - window_size_calc
        if(diff>1):
            buff_res = (window_size_dna - window_size_calc)//2
        else:
            buff_res = 0
        all_avgs = np.zeros(shape=(target_length, len(labels)))
        for i, hist_mark in enumerate(labels):
            if return_arcsinh:
                all_avgs[:, i] = np.arcsinh(np.mean(
                    np.nan_to_num(avg_data_conn[hist_mark].values(
                        the_chr, strt+buffer_bp+buff_res,
                        strt + window_size_calc - buffer_bp+buff_res,
                        numpy=True)
                    ).reshape(-1, pred_resolution),#averaging at desired res  
                            axis=1))
            else:
                all_avgs[:, i] = np.mean(
                    np.nan_to_num(avg_data_conn[hist_mark].values(
                        the_chr, strt+buffer_bp+buff_res,
                        strt + window_size_calc - buffer_bp+buff_res,
                        numpy=True)
                    ).reshape(-1, pred_resolution),#averaging at desired res  
                            axis=1)
        if include_atac:
            #have atac for each ouput track
            atac_dat = np.zeros(shape=(target_length, len(labels)))
            for i, hist_mark in enumerate(labels):
                if return_arcsinh:
                    atac_dat[:, i] = np.arcsinh(np.mean(
                        np.nan_to_num(atac_conn[cell]['atac'].values(
                            the_chr, strt+buffer_bp+buff_res,
                            strt + window_size_calc - buffer_bp+buff_res,
                            numpy=True)
                        ).reshape(-1, pred_resolution),#averaging at desired res  
                                axis=1))
                else:
                    atac_dat[:, i] = np.mean(
                        np.nan_to_num(atac_conn[cell]['atac'].values(
                            the_chr, strt+buffer_bp+buff_res,
                            strt + window_size_calc - buffer_bp+buff_res,
                            numpy=True)
                        ).reshape(-1, pred_resolution),#averaging at desired res  
                                axis=1)
                    
        X,y = next(generate_data(cells=cell,chromosomes=CHROMOSOMES,
                             cell_probs=np.repeat(1, 1),
                             chromosome_probs=np.repeat(1, len(CHROMOSOMES)),
                             features=features,labels=labels,data=data_conn,
                             window_size=window_size_dna,pred_res=pred_resolution,
                             pred_prop=pred_prop,rand_pos=False,
                             arcsin_trans = return_arcsinh,
                             chro=the_chr,pos=strt,
                             n_genomic_positions=window_size_CA,
                             training=False,
                             rtn_y_avg = False,
                             data_trans = data_trans_model))
        X_all.append(X['chrom_access_lcl'])
        #predict
        output = model.predict(X,return_arcsinh =return_arcsinh)
        #save pred, avg and actual
        model_pred.append(output.squeeze())
        avg_pred.append(all_avgs)
        signal.append(y.numpy().squeeze())
        if include_atac:
            atac.append(atac_dat)
        #save performance
        #move by the number of predictions we made
        strt+=target_bp

    signal = np.concatenate(signal,axis=0) 
    avg_pred = np.concatenate(avg_pred,axis=0)
    model_pred = np.concatenate(model_pred,axis=0)
    end_plot_pos = strt_plot_pos + (pred_resolution*signal.shape[0])
    if include_atac:
        atac = np.concatenate(atac,axis=0) 
        return (signal, avg_pred, model_pred, strt_plot_pos, end_plot_pos, atac)                    
    
    return (signal, avg_pred, model_pred, strt_plot_pos, end_plot_pos, X_all)                    
                    
                    
                    
                    
                    
def plot_tracks(tracks1, tracks2, tracks3 = None, 
                the_chr='', strt_bp = '', end_bp='',cell='', 
                labels = ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3'],
                tracks4 = None, 
                height=1, same_y = True, 
                save_plot = None, overall_title = None,
                nme_p1='Experiment',nme_p2='Average',
                nme_p3='Enformer Celltyping',
                nme_p4='Chromatin Accessibility',
                pal = None,
                figsize=(18., 16.)):
    """
    Plot predictions from actual, average and model
    """
    extended_palette = pal
    if pal is None:
        extended_palette = ["#9A8822","#F5CDB4","#F8AFA8",
                            "#FDDDA0","#74A089","#85D4E3",
                            #added extra to make 7
                            '#78A2CC']
    
    plt.figure(figsize=figsize)
    plt.rc('ytick', labelsize=16)
    
    num_rows = len(labels)*2
    num_cols = 1

    row_height = 4
    space_height = 4

    num_sep_rows = lambda x: int((x-1)/2)
    if tracks3 is not None:
        num_rows = len(labels)*3
        num_sep_rows = lambda x: int((x-1)/3)
    if tracks4 is not None:
        num_rows = len(labels)*4
        num_sep_rows = lambda x: int((x-1)/4)
        
    grid = (row_height*num_rows + space_height*num_sep_rows(num_rows), num_cols)

    ax = []

    for ind_row in range(num_rows):
        grid_row = row_height*ind_row + space_height*num_sep_rows(ind_row+1)

        ax += [plt.subplot2grid(grid, (grid_row, 0), rowspan=row_height)]

    plt.subplots_adjust(bottom=.05, top=.95, hspace=.1)
    
    plt.rc('axes', titlesize=14, labelsize=13)
    
    y_limits = [4.9, 4.9, 4.9, 4.9, 4.9, 4.9]#[4, 4, 4, 4, 4, 4]
    i = 0
    #two tracks    
    if tracks3 is None:
        for y_limit, title, y, y2 in zip(y_limits, labels, tracks1.T, tracks2.T):
            p1 = ax[i].fill_between(np.linspace(0, len(y), num=len(y)), y, 
                                    color=extended_palette[0])
            p2 = ax[i+1].fill_between(np.linspace(0, len(y2), num=len(y2)), y2,
                                      color=extended_palette[1])

            if same_y:
                ax[i].set_ylim([0, y_limit])
                ax[i+1].set_ylim([0, y_limit])

            ax[i].set_title(title, fontsize=15)
            ax[i+1].set_xticklabels([])
            sns.despine(top=True, right=True, bottom=True)
            i += 2
        ax[0].legend(
            [p1, p2],
            [nme_p1, nme_p2],
            bbox_to_anchor=(0.62, 1.6, 0.5, 0.5),
            fontsize=12
        )
    #three tracks
    elif tracks4 is None:
        for y_limit, title, y, y2, y3 in zip(y_limits, labels, tracks1.T, 
                                             tracks2.T, tracks3.T):
            p1 = ax[i].fill_between(np.linspace(0, len(y), num=len(y)), y, 
                                    color=extended_palette[0])
            p2 = ax[i+1].fill_between(np.linspace(0, len(y2), num=len(y2)), y2,
                                      color=extended_palette[1])
            p3 = ax[i+2].fill_between(np.linspace(0, len(y3), num=len(y3)), y3, 
                                      color=extended_palette[2])

            if same_y:
                ax[i].set_ylim([0, y_limit])
                ax[i+1].set_ylim([0, y_limit])
                ax[i+2].set_ylim([0, y_limit])

            ax[i].set_title(title, fontsize=15)
            ax[i+2].set_xticklabels([])
            sns.despine(top=True, right=True, bottom=True)
            i += 3
        ax[0].legend(
            [p1, p2, p3],
            [nme_p1, nme_p2, nme_p3],
            bbox_to_anchor=(0.62, 1.6, 0.5, 0.5),
            fontsize=12
        )
    else: #add atac too
        for y_limit, title, y, y2, y3, y4 in zip(y_limits, labels, tracks1.T, 
                                             tracks2.T, tracks3.T, tracks4.T):
            p1 = ax[i].fill_between(np.linspace(0, len(y), num=len(y)), y, 
                                    color=extended_palette[0])
            p2 = ax[i+1].fill_between(np.linspace(0, len(y2), num=len(y2)), y2,
                                      color=extended_palette[1])
            p3 = ax[i+2].fill_between(np.linspace(0, len(y3), num=len(y3)), y3, 
                                      color=extended_palette[2])
            p4 = ax[i+3].fill_between(np.linspace(0, len(y4), num=len(y4)), y4, 
                                      color=extended_palette[3])

            if same_y:
                ax[i].set_ylim([0, y_limit])
                ax[i+1].set_ylim([0, y_limit])
                ax[i+2].set_ylim([0, y_limit])
                ax[i+3].set_ylim([0, y_limit])

            ax[i].set_title(title, fontsize=15)
            ax[i+3].set_xticklabels([])
            sns.despine(top=True, right=True, bottom=True)
            i += 4
        ax[0].legend(
            [p1, p2, p3, p4],
            [nme_p1, nme_p2, nme_p3,nme_p4],
            bbox_to_anchor=(0.63, 2, 0.5, 0.5),
            fontsize=12
        )
    
    
    if (the_chr!='' or strt_bp != '' or end_bp!='' or cell!=''):
        plt.xlabel(f"{cell.capitalize()} {the_chr.capitalize()} :{strt_bp:,}-{end_bp:,}", 
               fontsize=15)
    if overall_title is not None:
        plt.suptitle(overall_title,fontsize=18)
    
    if save_plot is not None:
        plt.savefig(save_plot,bbox_inches='tight')


def plot_signal(track, labels, 
                the_chr, strt_bp, end_bp,cell, 
                height=1, same_y = True, 
                save_plot = None, overall_title = None):
    """
    Plot histone mark tracks
    """
    extended_palette = ["#9A8822","#F5CDB4","#F8AFA8",
                    "#FDDDA0","#74A089","#85D4E3",
                    #added extra to make 7
                    '#78A2CC']
    
    plt.figure(figsize=(16., 12.))
    plt.rc('ytick', labelsize=16)
    
    num_rows = len(labels)
    num_cols = 1

    row_height = 4
        
    grid = (row_height*num_rows,num_cols)

    ax = []

    for ind_row in range(num_rows):
        grid_row = row_height*ind_row#*num_sep_rows(ind_row+1)

        ax += [plt.subplot2grid(grid, (grid_row, 0), rowspan=row_height)]

    #plt.subplots_adjust(bottom=.05, top=.95, hspace=.1)
    
    plt.rc('axes', titlesize=13, labelsize=13)
    
    y_limits = [4.9, 4.9, 4.9, 4.9, 4.9, 4.9]#[4, 4, 4, 4, 4, 4]
    i = 0
    p = []
    for y_limit, title, y in zip(y_limits, labels, track[0].T):
        p.append(ax[i].fill_between(np.linspace(0, len(y), 
                                                num=len(y)), 
                                    y, color=extended_palette[i]))

        if same_y:
            ax[i].set_ylim([0, y_limit])

        #ax[i].set_title(title, fontsize=13,pad=.5)
        ax[i].text(0.5, 0.8, title,size=15,
                   transform=ax[i].transAxes, ha="center")
        ax[i].set_xticklabels([])
        sns.despine(top=True, right=True, bottom=True)
        i += 1
    ax[0].legend(
        p,
        labels,
        bbox_to_anchor=(0.62, 1, 0.5, 0.5),
        fontsize=12
    )
    
    
    plt.xlabel(f"{cell.capitalize()} {the_chr.capitalize()} :{strt_bp:,}-{end_bp:,}", 
               fontsize=15)
    if overall_title is not None:
        plt.suptitle(overall_title,fontsize=18)
    
    if save_plot is not None:
        plt.savefig(save_plot,bbox_inches='tight')

def plot_signal(tracks, interval, height=1.5,
                pal=["#9A8822","#F5CDB4","#F8AFA8",
                    "#FDDDA0","#74A089","#85D4E3"]):
    """
    Simple plot function for the output of the Enformer
    Celltyping prediction or another similar signal.
    """
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    i=0
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval['start'], 
                                    interval['end'], 
                                    num=len(y)), y,
                       color=pal[i])
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
        i+=1
    
    ax.set_xlabel(f"{interval['cell']} {interval['chro']}-{interval['start']}:{interval['end']}")
    plt.tight_layout()
        
def measure_receptive_field(model, seq_length: int = 196_608,
                            window_size_lcl: int = 1562*128,
                            window_size_gbl: int = 1216*3_000,
                            N_iter: int = 100,N_pos: int = 100,
                            model_name: str = 'Enformer_Celltyping',
                            enf_tracks: list = list(range(5313)),
                            agg_snp_eff_centre_bp: int = 128*4,
                            pred_res: int = 128
                           ):
    """
    Measure the receptive field of a model by running the
    mutation experiment at several locations in the sequence.
    
    Arguments: 
    model:
        Model to calculate the receptive field for.
    seq_length:
        Length of DNA input the model takes.
    window_size_lcl:
        Base-pairs of chromatin accessibility data used
        to embed local cell type information. Only relevant 
        for Enformer Celltyping.
    window_size_gbl:
        Base-pairs of chromatin accessibility data used
        to embed global cell type information. Only relevant 
        for Enformer Celltyping.
    N_iter:
        Number of iterations for receptive field test at each 
        positions.
    N_pos:
        Number of positions to test receptive field, evenly 
        spaced across the input sequence length
    model_name:
        Model to be checked, can be one of Enformer or 
        Enformer_Celltyping
    enf_tracks:
        For enformer model which tracks to check for mutational
        differences. Might be appropriate to remove expression/
        TF tracks. Default is all tracks.
    agg_snp_eff_centre_bp:
        The centre number of base-pairs to aggregate the effect of 
        a SNP on. Not using all predicted base-pairs as we want to
        be sure SNP's at the edge of the input window have long range 
        effects on the centre positions. Default 512
    pred_res:
        Predictive resolution of the model. Default is 128.
    """

    mutation_locations = np.linspace(0, seq_length - 1, N_pos)
    avg_delta_pred_all = []
    avg_delta_mut_all = []
    for mutation_location in tqdm(mutation_locations):
        mutation_location = int(mutation_location)
        if (model_name.lower()=='enformer_celltyping'):
            avg_delta_pred, avg_delta_mut = _run_mut_enf_celltyping(model,
                                                                    mutation_location,
                                                                    seq_length,
                                                                    window_size_lcl,
                                                                    window_size_gbl,
                                                                    N_iter,
                                                                    agg_snp_eff_centre_bp,
                                                                    pred_res
                                                                   )
        elif (model_name.lower()=='enformer'):
            avg_delta_pred, avg_delta_mut = _run_mut_enformer(model,
                                                              mutation_location,
                                                              seq_length,
                                                              N_iter,
                                                              enf_tracks,
                                                              agg_snp_eff_centre_bp
                                                             )   
        else:
            raise ValueError(f'The model {model_name} is not supported.')
            
        avg_delta_pred_all.append(avg_delta_pred)
        avg_delta_mut_all.append(avg_delta_mut)
    #make dictionary with postion and mean change
    delta_pred = dict(zip(mutation_locations, avg_delta_pred_all))
    delta_mut = avg_delta_mut_all
    
    return(delta_pred,delta_mut)


def _run_mut_enf_celltyping(model, mutation_pos: int,
                            seq_length: int = 196_608,
                            window_size_lcl: int = 1562*128,
                            window_size_gbl: int = 1216*3_000,
                            N_iter: int = 100,
                            agg_snp_eff_centre_bp: int = 128*4,
                            pred_res: int = 128
                           ):
    """
    Measure the receptive field of the Enformer Celltyping model by:
    (1) predicting on a random sequence of DNA
    (2) mutate one base pair
    (3) predict on the new sequence of DNA
    (4) measure the expression difference
    (5) repeat steps 1-4 multiple times, then take the average
    Arguments:
    model:
        Model to calculate the receptive field for.
    mutation_pos:
        Location in the sequence for the mutation - integer
    seq_length:
        Length of DNA input the model takes.
    window_size_lcl:
        Base-pairs of chromatin accessibility data used
        to embed local cell type information.
    window_size_gbl:
        Base-pairs of chromatin accessibility data used
        to embed global cell type information.
    N_iter:
        Number of iterations for receptive field test at each 
        positions.
    agg_snp_eff_centre_bp:
            The centre number of base-pairs to aggregate the effect of 
            a SNP on. Not using all predicted base-pairs as we want to
            be sure SNP's at the edge of the input window have long range 
            effects on the centre positions. Default 512
    pred_res:
        Predictive resolution of the model. Default is 128.
    Returns:
        The average change in the prediction for the model by mutation 
        and position
    """
    BASE_PAIRS = np.eye(4)
    changed_pred = []

    for i in range(N_iter):
        # Baseline run -----------------
        random_dna = BASE_PAIRS[
            np.random.choice(BASE_PAIRS.shape[0], size=seq_length)
        ][np.newaxis, :, :]
        #all raw input data will be averaged at 25bp - output from MACS2
        rand_chro_access_lcl = np.array(np.random.random((1, window_size_lcl//pred_res)), 
                                        dtype=np.float32)
        rand_chro_access_gbl = np.array(np.random.random((1, window_size_gbl//250)), 
                                        dtype=np.float32)
        rand_x = {"dna":random_dna,
                  "chrom_access_lcl":rand_chro_access_lcl,
                  "chrom_access_gbl":rand_chro_access_gbl}
        #predict
        baseline_pred = model.predict(rand_x,
                                      return_arcsinh = False)

        # Mutation run ------------------
        mutated_dna = copy.deepcopy(random_dna)
        mutated_dna[0, mutation_pos, :] = np.roll(
            mutated_dna[0, mutation_pos, :],
            shift=np.random.randint(1, 3)
        )
        #chrom access stays the same
        rand_mut_x = {"dna":mutated_dna,
                      "chrom_access_lcl":rand_chro_access_lcl,
                      "chrom_access_gbl":rand_chro_access_gbl}
        #predict
        mut_pred = model.predict(rand_mut_x,
                                 return_arcsinh = False)

        # Measure the difference in expression level
        difference = mut_pred - baseline_pred
        #save the difference
        changed_pred.append(difference)
    
    #aggregate - get the average across pred positions
    avg_changed_pred = np.mean(np.mean(np.abs(changed_pred), 
                                            axis=0),#avg across N_iters 
                                    axis=2)[0]#avg across output tracks 
    #aggregate - get the average change by mutational positions
    #NOTE enf celltyping preds on 0.114 of input DNA's 196_608bp's enf does 0.5833333
    #need to adjust enf avg to just 0.114 rather than 0.58333
    #in fact we should only be testing the change on the very centre to make sure 
    #long distance SNPs are taken into account
    target_bp = agg_snp_eff_centre_bp//pred_res
    target_pos = len(avg_changed_pred)
    buffer = (target_pos-target_bp)//2
    avg_changed_mut = np.mean(avg_changed_pred[buffer-1:buffer+target_bp-1])
    return avg_changed_pred, avg_changed_mut

def _run_mut_enformer(model, mutation_pos: int,
                      seq_length: int = 196_608,
                      N_iter: int = 100,
                      enf_tracks: list = list(range(5313)),
                      agg_snp_eff_centre_bp: int = 128*4#math.lcm(25,128)#3200
                     ):
    """
    Measure the receptive field of the Enformer model by:
    (1) predicting on a random sequence of DNA
    (2) mutate one base pair
    (3) predict on the new sequence of DNA
    (4) measure the expression difference
    (5) repeat steps 1-4 multiple times, then take the average
    Arguments:
        model:
            Model to calculate the receptive field for.
        mutation_pos:
            Location in the sequence for the mutation - integer
        seq_length:
            Length of DNA input the model takes.
        N_iter:
            Number of iterations for receptive field test at each 
            position.
        agg_snp_eff_centre_bp:
            The centre number of base-pairs to aggregate the effect of 
            a SNP on. Not using all predicted base-pairs as we want to
            be sure SNP's at the edge of the input window have long range 
            effects on the centre positions. Default is 512.
    Returns:
        The average change in the prediction for the model by mutation 
        and position
    """
    BASE_PAIRS = np.eye(4)
    changed_pred = []

    for i in range(N_iter):
        # Baseline run -----------------
        random_dna = BASE_PAIRS[
            np.random.choice(BASE_PAIRS.shape[0], size=seq_length)
        ][np.newaxis, :, :]
        # Enformer accepts a longer input sequence so the ends need to be padded
        pad_enformer = np.zeros(shape=(1, seq_length // 2, 4))
        random_dna_enformer = np.concatenate((pad_enformer, random_dna, 
                                              pad_enformer), axis=1)
        #predict
        baseline_pred = model.predict_on_batch(random_dna_enformer)['human'][0].numpy()
        #filter to the tracks of interest
        baseline_pred = baseline_pred[:,enf_tracks]
        # Mutation run ------------------
        mutated_dna = copy.deepcopy(random_dna)
        mutated_dna[0, mutation_pos, :] = np.roll(
            mutated_dna[0, mutation_pos, :],
            shift=np.random.randint(1, 3)
        )
        mutated_dna_enformer = np.concatenate((pad_enformer, mutated_dna, 
                                               pad_enformer), axis=1)
        #predict
        mut_pred = model.predict_on_batch(mutated_dna_enformer)['human'][0].numpy()
        #filter to the tracks of interest
        mut_pred = mut_pred[:,enf_tracks]
        
        # Measure the difference in expression level
        difference = mut_pred - baseline_pred
        #save the difference
        changed_pred.append(difference)
    
    #aggregate - get the average across pred positions
    avg_changed_pred = np.mean(np.mean(np.abs(changed_pred), 
                                            axis=0),#avg across N_iters 
                                    axis=1)#avg across output tracks 
    #aggregate - get the average change by mutational positions
    #NOTE enf celltyping preds on 0.114 of input DNA's 196_608bp's enf does 0.5833333
    #need to adjust enf avg to just 0.114 rather than 0.58333
    target_bp = int((0.114*seq_length)//128)
    target_pos = len(avg_changed_pred)
    buffer = (target_pos-target_bp)//2
    avg_changed_pred_org = avg_changed_pred.copy()
    avg_changed_pred = avg_changed_pred[buffer-1:buffer+target_bp-1]
    #in fact we should only be testing the change on the very centre to make sure 
    #long distance SNPs are taken into account
    target_bp = agg_snp_eff_centre_bp//128
    target_pos = len(avg_changed_pred)
    buffer = (target_pos-target_bp)//2
    avg_changed_mut = np.mean(avg_changed_pred[buffer-1:buffer+target_bp-1])
    return avg_changed_pred_org, avg_changed_mut 



def create_ref_alt_DNA_window(chro: str, pos: int,
                              window_size_dna: int = 196_608,
                              window_size_CA: int = 1562*128,
                              pred_prop: float = (128*896)/196_608,#0.113,
                              pred_resolution: int = 128):#25):
    """
    Get the start positions for the DNA sequences to pass to 
    model data generator to cover window size of predicitons
    Arguments:
        chro:
            Chromosome for DNA sequence. Should be formatted as
            chr1-22.
        pos:
            Base-pair position of the ref (A1) and alt (A2) 
            base-pairs.
        window_size_dna: 
            Window size of DNA input for the model. Default is 
            196_608 - Enformer's DNA window size.
        window_size_CA: 
            Window size of chromatin accessibility input for the model. 
            Default is 1562*128 - Enformer Celltyping's local chromatin
            accessibility window size.    
        pred_prop:
            Centred proportion of base-pairs that the model predicts.
            This is necessary since these models predict in a funnel
            style in that every position has a buffer of DNA on 
            either side to make the prediction. Default is 
            (128*896)/196_608.
        pred_resolution:
            The resolution (number of base-pairs averaged) at which the 
            model predicts. Default is 128bp, Enformer Celltyping's 
            predicted resolution.
    returns:
        start positions for DNA sequence and SNP position relative to the
        DNA start position so predicitons are made for the full window 
        size based on the SNP.
    """
    #Get limits for chromosome
    chromosome_len = CHROMOSOME_DATA[chro]
    #can only predict if SNP can be centred
    #without hitting start or end of chrom
    window_size_max = max([window_size_dna,window_size_CA])
    if((pos - window_size_max//2)<0 or (pos + window_size_max//2)>chromosome_len):
        #return empty
        print(f"SNP {chro}:{pos} is too close to chromosome end/start to predict.")
        return [], []
    else:
        #rest uses dna window size
        window_size = window_size_dna
        #calc the actual base-pairs
        buffer_bp,target_length,target_bp = create_buffer(window_size=window_size,
                                                          pred_res=pred_resolution,
                                                          pred_prop= pred_prop)
        #need to get all the DNA positions to fill full receptive field
        pred_window = 0
        abs_snp_pos = 0
        buff = 0
        dna_strt = []
        snp_pos = []
        dna_strt_up = []
        snp_pos_up = []
        dna_strt_down = []
        snp_pos_down = []
        #since CA could be wider than DNA window need to work out diff
        #will be 0 if same size or dna bigger
        CA_dna_bp_diff = window_size_max - window_size_dna
        while(pred_window<window_size):
            #first one is centred
            if(pred_window == 0):
                dna_strt.append(pos - window_size//2)
                #calc snp pos from start of dna
                snp_pos.append(window_size//2)
                #set both as DNA start
                upstream_pos = pos - window_size//2#pos
                downstream_pos = pos - window_size//2#pos
                pred_window = pred_window + target_bp
            #get last on either end too so we can move in by
            #buffer amount to avoid spilling outside input window
            elif(pred_window+(target_bp*2)>window_size):
                #calc buffer so shift isn't by full target bp amount
                minus_buff = (pred_window+(target_bp*2)-window_size)//2
                abs_snp_pos = abs_snp_pos + target_bp - minus_buff
                upstream_pos = upstream_pos - target_bp + minus_buff
                downstream_pos = downstream_pos + target_bp -minus_buff
                #upstream
                #have to check it is within the chrom
                #take in a/c extra range of CA if there is one
                if((upstream_pos - (CA_dna_bp_diff//2) >0)):
                    dna_strt_up.append(upstream_pos)# - window_size//2)
                    #center plus amt moved
                    snp_pos_up.append((window_size//2) + abs_snp_pos)
                #downstream
                #have to check it is within the chrom
                #take in a/c extra range of CA if there is one
                if((downstream_pos+window_size+(CA_dna_bp_diff//2))<=chromosome_len):
                    dna_strt_down.append(downstream_pos) #- window_size//2)
                    #center minus amt moved
                    snp_pos_down.append((window_size//2) - abs_snp_pos)
                #still increment even if at the edge so it doesn't increase too 
                #much in one direction and move SNP outside of input window
                pred_window = pred_window + (target_bp*2) - (minus_buff*2)
            #neither first or last    
            else:    
                abs_snp_pos = abs_snp_pos + target_bp
                upstream_pos = upstream_pos - target_bp
                downstream_pos = downstream_pos + target_bp 
                #upstream
                #have to check it is within the chrom
                if((upstream_pos >0)):
                    dna_strt_up.append(upstream_pos)# - window_size//2)
                    #center plus amt moved
                    snp_pos_up.append((window_size//2) + abs_snp_pos)
                #downstream
                #have to check it is within the chrom
                if((downstream_pos+window_size)<=chromosome_len):
                    dna_strt_down.append(downstream_pos) #- window_size//2)
                    #center minus amt moved
                    snp_pos_down.append((window_size//2) - abs_snp_pos)
                #still increment even if at the edge so it doesn't increase too 
                #much in one direction and move SNP outside of input window
                pred_window = pred_window + (target_bp*2)
        #return dna_strt, snp_pos
        #reverse so in increasing order on chrom
        dna_strt_up.reverse()
        snp_pos_up.reverse()
        dna_strt = dna_strt_up+dna_strt+dna_strt_down
        snp_pos = snp_pos_up+snp_pos+snp_pos_down
        return dna_strt,snp_pos


from operator import add

def plot_snp_dna_window(dna_strt: list, snp_pos: list,
                        window_size_dna: int = 196_608,
                        window_size_CA: int = 1562*128,
                        pred_prop: float = (128*896)/196_608,
                        pred_resolution: int = 128):
    """
    Get the start positions for the DNA sequences to pass to 
    model data generator to cover window size of predicitons
    Arguments:
        dna_strt: 
            Start positions for DNA sequence so predicitons 
            are made for the full window size based on the 
            SNP.
        snp_pos:
            SNP position relative to the DNA start position so 
            predicitons are made for the full window size based 
            on the SNP.
        window_size_dna: 
            Window size of DNA input for the model. Default is 
            196_608 - Enformer's DNA window size.
        window_size_CA: 
            Window size of chromatin accessibility input for the model. 
            Default is 1562*128 - Enformer Celltyping's local chromatin
            accessibility window size.   
        pred_prop:
            Centred proportion of base-pairs that the model predicts.
            This is necessary since these models predict in a funnel
            style in that every position has a buffer of DNA on 
            either side to make the prediction. Default is 
            (128*896)/196_608.
        pred_resolution:
            The resolution (number of base-pairs averaged) at which the 
            model predicts. Default is 128bp, Enformer Celltyping's 
            predicted resolution.    
    returns:
        plot of the DNA positions predicted for.
    """
    #calc the actual base-pairs
    buffer_bp,target_length,target_bp = create_buffer(window_size=window_size_dna,
                                                      pred_res=pred_resolution,
                                                      pred_prop= pred_prop)
    
    #Calc relative data
    dna_end = [x+window_size_dna for x in dna_strt]
    snp_pos_real = list(map(add, dna_strt, snp_pos))
    pred_strt = [x+buffer_bp for x in dna_strt]
    pred_end = [x-buffer_bp for x in dna_end]
    tmp = pd.DataFrame({'dna_strt': dna_strt,
                    'dna_end': dna_end,
                    'snp_pos': snp_pos_real,
                    'pred_strt': pred_strt,
                    'pred_end' : pred_end})
    tmp["id"] = tmp.index
    tmp = pd.melt(tmp, id_vars='id', value_vars=['dna_strt', 'dna_end', 
                                                 'snp_pos','pred_strt',
                                                 'pred_end'])
    tmp['col'] = np.where(tmp['variable'].isin(['pred_strt','pred_end']), 
                          'Out', 'In')
    #update colour so snp different
    tmp['colour'] = np.where(tmp['variable'].isin(['snp_pos']), 
                             len(set(tmp['id'])), tmp['id'])
    g = sns.FacetGrid(tmp, col="id",hue="colour",col_wrap=5)
    #g = plt.scatter(y=tmp['value'], x=tmp['col'],c=tmp['id'])
    g.map(sns.scatterplot, 'col', 'value')
    # There is no labels, need to define the labels
    legend_labels  = ['position '+str(x//2) if (x%2==0) else 'SNP' for x in range(len(set(tmp['id']))*2)]
    g.add_legend(title='', labels=legend_labels)
    pos = snp_pos_real[0]
    g.map(plt.axhline, 
          y=pos - window_size_dna//2, 
          ls='--', c='red')
    g.map(plt.axhline, 
          y=pos + window_size_dna//2, 
          ls='--', c='red')
    return g
       
def predict_snp_effect_sldp(model, alt: str, cell: str, chro: str,
                            dna_strt: list, snp_pos: list,
                            data_generator: tf.data.Dataset,
                            effect_mode: str = 'both',
                            window_size_dna: int = 196_608,
                            window_size_CA: int = 1562*128,
                            pred_prop: float = (128*896)/196_608,
                            pred_resolution: int = 128):
    """
    Measure the effect of a SNP on the model's predictions:
    (1) Measure calculated peaks of model with ref allele
    (2) Measure calculated peaks of model with alt allele
    (3) Calculate the effective change caused by the SNP
    (4) All calcs will be done with rev comp and small rand perms
    (5) Aggregate so there is one effective change per SNP.
    
    NOTE - Enformer when conducting this, centred the SNP in DNA input. 
    This means they don't utilise the full 100kbp in either direction 
    since the prediction window is (896*128)/2 = 57344 in either direction.
    We don't want to fall into the same trap since our prediction window is 
    even smaller. To avoid this, we will centre input on SNP then gradually 
    move off centre so output window makes up the full input window size. 
    This is done in the create_ref_alt_DNA_window() function. The effective
    change of the resulting window is tested here.
    
    Arguments:
        model:
            Model to calculate the SNP effect on.
        cell:
            Cell ID to predict on, must match the data generator.
        alt:
            Alternative (A2) allele to measure the effect of. 
            Should be one of A,T,C,G and shouldn't match the 
            reference (A1) allele.
        chro:
            Chromosome for DNA sequence. Should be formatted as
            chr1-22.    
        dna_strt: 
            Start positions for DNA sequence so predicitons 
            are made for the full window size based on the 
            SNP.
        snp_pos:
            SNP position relative to the DNA start position so 
            predicitons are made for the full window size based 
            on the SNP.
        data_generator:
            Data generator to load data from the specific cell type
            to predict the SNP effect on. See generate_sample which
            is a wrapper for generate_data() for an example of an 
            appropriate data generator.
        effect_mode:
            Aggregating calculated effects can aggregate the bp 
            level effective change by sum or max for the QTL, 
            default is to do both.
        window_size_dna: 
            Window size of DNA input for the model. Default is 
            196_608 - Enformer's DNA window size.
        window_size_CA: 
            Window size of chromatin accessibility input for the model. 
            Default is 1562*128 - Enformer Celltyping's local chromatin
            accessibility window size.    
        pred_prop:
            Centred proportion of base-pairs that the model predicts.
            This is necessary since these models predict in a funnel
            style in that every position has a buffer of DNA on 
            either side to make the prediction. Default is 
            (128*896)/196_608.
        pred_resolution:
            The resolution (number of base-pairs averaged) at which the 
            model predicts. Default is 128bp, Enformer Celltyping's 
            predicted resolution.      
    Returns:
        The effective change in peaks related to the SNP aggregated for 
        the model's output channels.
    """
    
    #validate inputs
    assert alt in ['A','C','G','T'], "Alt must be one of 'A','C','G','T'"

    def effect_func_sum(a1,a2,axis=0):
        return np.sum(a1-a2,axis=axis)
    def absmax(a,b):
        return (np.where(np.abs(a) > np.abs(b), a, b))

    def effect_func_max(a1,a2,axis=0):
        a = np.max(a1-a2,axis=axis)
        b = np.min(a1-a2,axis=axis)
        return absmax(a,b)
    
    def effect_func_max_sum(a1,a2,axis=0):
        a = np.max(a1-a2,axis=axis)
        b = np.min(a1-a2,axis=axis)
        return(absmax(a,b),
               np.sum(a1-a2,axis=axis))

    effect_mode = effect_mode.lower()

    if effect_mode=='sum':
        effect_func = effect_func_sum
    elif effect_mode=='max':
        effect_func = effect_func_max
    elif effect_mode=='both':
        effect_func = effect_func_max_sum    
    assert effect_mode in ['sum','max','both'], 'Unknown effect function, use sum or max.'
    
    #calc the actual base-pairs
    buffer_bp,target_length,target_bp = create_buffer(window_size=window_size_dna,
                                                      pred_res=pred_resolution,
                                                      pred_prop= pred_prop)
    
    #get cell id from name
    cell_id = list(CELLS.keys())[list(CELLS.values()).index(cell)]

    #load reference data, predict ref and alt and concat
    wind_size_pred_ref = []
    wind_size_pred_alt = []
    for ind,strt_i in enumerate(dna_strt):
        pos_i = snp_pos[ind]
        ref_all = data_generator.load(pos=strt_i,chro=chro,cell=cell)
        rand_seq_shift_amt = ref_all[1]
        ref_seq = {"dna":ref_all[0]['dna'],
                   "chrom_access_gbl":ref_all[0]['chrom_access_gbl'],
                   "chrom_access_lcl":ref_all[0]['chrom_access_lcl']
                  }
        #Now impute SNP for alt
        one_hot_alt = one_hot_encode_dna(alt)
        one_hot_alt_rc = rev_comp_dna(one_hot_alt,one_hot=True)
        #add in alt allele
        #can't update tensor obj so convert back to np
        np_alt_dna = ref_seq['dna'].numpy()
        #original sequence
        np_alt_dna[0,pos_i,:] = one_hot_alt
        #rand perm
        np_alt_dna[1,pos_i-rand_seq_shift_amt,:] = one_hot_alt
        #rev comp
        np_alt_dna[2,ref_seq['dna'].shape[1]-pos_i,:] = one_hot_alt_rc
        #rev comp of rand perm
        np_alt_dna[3,ref_seq['dna'].shape[1]-pos_i+rand_seq_shift_amt,:] = one_hot_alt_rc
        alt_seq = {"dna":tf.convert_to_tensor(np_alt_dna),
                   "chrom_access_gbl":ref_seq['chrom_access_gbl'],
                   "chrom_access_lcl":ref_seq['chrom_access_lcl']
                  }
        #Now both ref and alt are ready to be passed to model
        pred_ref = model.predict(ref_seq,return_arcsinh = False)
        pred_alt = model.predict(alt_seq,return_arcsinh = False)
        #edge prediction region results may need to be chopped if overlap previous
        #this will be the case when the target bp region isn't a mutliple of the input 
        #region
        #this will be the first and last region - NOTE dna start positions must be in order
        if(ind==0 and dna_strt[ind+1]-dna_strt[ind]<target_bp):
            #first so chop end
            #remember 3 and 4 are rev comp so take from start not end
            chop_bp = target_bp-(dna_strt[ind+1]-dna_strt[ind])
            chop_pos = chop_bp//pred_resolution
            pred_ref = np.concatenate((pred_ref[0:2,0:target_length-chop_pos,:], 
                                       pred_ref[2:4,chop_pos:target_length,:]), 
                                      axis=0)
            pred_alt = np.concatenate((pred_alt[0:2,0:target_length-chop_pos,:], 
                                       pred_alt[2:4,chop_pos:target_length,:]), 
                                      axis=0)
        elif(ind==len(dna_strt)-1 and dna_strt[ind]-dna_strt[ind-1]<target_bp):    
            #last so chop start
            #remember 3 and 4 are rev comp so take from start not end
            chop_bp = target_bp-(dna_strt[ind]-dna_strt[ind-1])
            chop_pos = chop_bp//pred_resolution
            pred_ref = np.concatenate((pred_ref[0:2,chop_pos:target_length,:], 
                                       pred_ref[2:4,0:target_length-chop_pos,:]), 
                                      axis=0)
            pred_alt = np.concatenate((pred_alt[0:2,chop_pos:target_length,:], 
                                       pred_alt[2:4,0:target_length-chop_pos,:]), 
                                      axis=0)
        wind_size_pred_ref.append(pred_ref)
        wind_size_pred_alt.append(pred_alt)
    #combine to one sequence for all of the window size
    wind_size_pred_ref = tf.concat(wind_size_pred_ref,axis=1)
    wind_size_pred_alt = tf.concat(wind_size_pred_alt,axis=1)
    #Now aggregate the SNP effect based on chosen method
    #loop for each of org, rand perm, rev comp and average
    if effect_mode=='both':
        eff_max = []
        eff_sum = []
        for i in range(wind_size_pred_ref.shape[0]):
            max_i,sum_i = effect_func(wind_size_pred_ref[i,:,:],wind_size_pred_alt[i,:,:])
            eff_max.append(max_i)
            eff_sum.append(sum_i)
        agg_eff_max = np.mean(np.vstack(eff_max),axis=0)    
        agg_eff_sum = np.mean(np.vstack(eff_sum),axis=0)
        return(agg_eff_max,agg_eff_sum)
    else:    
        eff=[]        
        for i in range(wind_size_pred_ref.shape[0]):
            eff.append(effect_func(wind_size_pred_ref[i,:,:],wind_size_pred_alt[i,:,:]))
        agg_eff = np.mean(np.vstack(eff),axis=0)
        return(agg_eff)


def predict_snp_effect_sldp_checkpoint(model, alt: str, cell: str, chro: str,
                            dna_strt: list, snp_pos: list,
                            data_generator: tf.data.Dataset,
                            checkpoint_pth: str,#or list
                            no_pred: bool = False,
                            effect_mode: str = 'both',
                            window_size_dna: int = 196_608,
                            window_size_CA: int = 1562*128,
                            pred_prop: float = (128*896)/196_608,
                            pred_resolution: int = 128):
    """
    Measure the effect of a SNP on the model's predictions:
    (1) Measure calculated peaks of model with ref allele
    (2) Measure calculated peaks of model with alt allele
    (3) Calculate the effective change caused by the SNP
    (4) All calcs will be done with rev comp and small rand perms
    (5) Aggregate so there is one effective change per SNP.
    
    NOTE - Enformer when conducting this, centred the SNP in DNA input. 
    This means they don't utilise the full 100kbp in either direction 
    since the prediction window is (896*128)/2 = 57344 in either direction.
    We don't want to fall into the same trap since our prediction window is 
    even smaller. To avoid this, we will centre input on SNP then gradually 
    move off centre so output window makes up the full input window size. 
    This is done in the create_ref_alt_DNA_window() function. The effective
    change of the resulting window is tested here.
    
    Differs to predict_snp_effect_sldp as it converts DNA in the data loader
    and saves ref and alt input as checkpoints so it doesn't need to be rerun
    when ona. new cell type. Slower up front but quicker when looking at more
    than 1 cell type per sumstats (or multi sumstats).
    
    Arguments:
        model:
            Model to calculate the SNP effect on.
        cell:
            Cell Name to predict on, must match the data generator.
        alt:
            Alternative (A2) allele to measure the effect of. 
            Should be one of A,T,C,G and shouldn't match the 
            reference (A1) allele.
        chro:
            Chromosome for DNA sequence. Should be formatted as
            chr1-22.    
        dna_strt: 
            Start positions for DNA sequence so predicitons 
            are made for the full window size based on the 
            SNP.
        snp_pos:
            SNP position relative to the DNA start position so 
            predicitons are made for the full window size based 
            on the SNP.
        data_generator:
            Data generator to load data from the specific cell type
            to predict the SNP effect on. See generate_sample which
            is a wrapper for generate_data() for an example of an 
            appropriate data generator.
        checkpoint_pth:
            Where to look for and save loaded files to speed up SNP
            effect predicitions. Can handle one or multiple file paths.
            If multiple past, first will be used for future checkpoint
            saves.
        no_pred:
            Bool indicating whether predictions should be made, if not 
            the function is only useful to save the positions in the
            checkpoint folder to speed up future runs.
        effect_mode:
            Aggregating calculated effects can aggregate the bp 
            level effective change by sum or max for the QTL, 
            default is to sum.
        window_size_dna: 
            Window size of DNA input for the model. Default is 
            196_608 - Enformer's DNA window size.
        window_size_CA: 
            Window size of chromatin accessibility input for the model. 
            Default is 1562*128 - Enformer Celltyping's local chromatin
            accessibility window size.
        pred_prop:
            Centred proportion of base-pairs that the model predicts.
            This is necessary since these models predict in a funnel
            style in that every position has a buffer of DNA on 
            either side to make the prediction. Default is 
            (128*896)/196_608.
        pred_resolution:
            The resolution (number of base-pairs averaged) at which the 
            model predicts. Default is 128bp, Enformer Celltyping's 
            predicted resolution.   
    Returns:
        The effective change in peaks related to the SNP aggregated for 
        the model's output channels.
    """
    
    #validate inputs
    assert alt in ['A','C','G','T'], "Alt must be one of 'A','C','G','T'"

    def effect_func_sum(a1,a2,axis=0):
        return np.sum(a1-a2,axis=axis)
    def absmax(a,b):
        return (np.where(np.abs(a) > np.abs(b), a, b))

    def effect_func_max(a1,a2,axis=0):
        a = np.max(a1-a2,axis=axis)
        b = np.min(a1-a2,axis=axis)
        return absmax(a,b)
    
    def effect_func_max_sum(a1,a2,axis=0):
        a = np.max(a1-a2,axis=axis)
        b = np.min(a1-a2,axis=axis)
        return(absmax(a,b),
               np.sum(a1-a2,axis=axis))
    
    def load_gbl_atac_ref(link):
        """
        sometimes when running mutliple in parallel, multiple
        scripts referencing the one np file causes an error.
        Use try catch to instead load a copy of the file if this
        happens.
        """
        try:
            dat = np.load(DATA_PATH/"sldp/checkpoint/CD4T_ATAC1.npz")
        except:
            #get link of copied version
            link_copy = os.path.splitext(link)[0]+'_2.npz'
            print(f"Failed to load ref global atac,trying copy {link_copy}")
            dat = np.load(link_copy)
        return(dat)    

    effect_mode = effect_mode.lower()

    if effect_mode=='sum':
        effect_func = effect_func_sum
    elif effect_mode=='max':
        effect_func = effect_func_max
    elif effect_mode=='both':
        effect_func = effect_func_max_sum    
    assert effect_mode in ['sum','max','both'], 'Unknown effect function, use sum, max or both.'
    
    #calc the actual base-pairs
    buffer_bp,target_length,target_bp = create_buffer(window_size=window_size_dna,
                                                      pred_res=pred_resolution,
                                                      pred_prop= pred_prop)
    
    #get cell id from name
    cell_id = list(CELLS.keys())[list(CELLS.values()).index(cell)]
    
    #if user inputs one checkpoint path as str, make a list
    if isinstance(checkpoint_pth, str):
        checkpoint_pth = [checkpoint_pth]

    #load reference data, predict ref and alt and concat
    ref_seq_dna_all = []
    ref_seq_prom_all = []
    ref_seq_lcl_all = []
    alt_seq_dna_all = []
    alt_seq_prom_all = []
    alt_seq_lcl_all = []
    for ind,strt_i in enumerate(dna_strt):
        #calc local chrom access start from dna start
        lcl_CA_strt_i = strt_i + ((window_size_dna-window_size_CA)//2)
        pos_i = snp_pos[ind]
        ref_pth = [i+f'/{chro}_{strt_i}.npz' for i in checkpoint_pth]
        ref_atac_pth = [i+f'/{cell_id}_ATAC.npz' for i in checkpoint_pth]
        ref_atac_lcl_pth = [i+f'/{lcl_CA_strt_i}_ATAC_lcl.npz' for i in checkpoint_pth]
        alt_pth = [i+f'/{chro}_{strt_i}_{alt}_{pos_i}.npz' for i in checkpoint_pth]
        any_ref = any([os.path.isfile(x) for x in ref_pth])
        any_ref_atac = any([os.path.isfile(x) for x in ref_atac_pth])
        any_ref_atac_lcl = any([os.path.isfile(x) for x in ref_atac_lcl_pth])
        #only generate if not saved
        if (not any_ref) or (not any_ref_atac) or (not any_ref_atac_lcl):
            #DNA
            if not any_ref:
                #don't bother loading chrom access - speed
                ref_all = data_generator.load(pos=strt_i,chro=chro,cell=cell,
                                              return_chrom_access=False)
                rand_seq_shift_amt = ref_all[1]
                ref_seq = ref_all[0]
                #save for next time
                np.savez(ref_pth[0],dna=ref_seq,rand_seq_shift_amt=rand_seq_shift_amt)
            # local & global chromatin accessibility signature
            if (not any_ref_atac_lcl) or (not any_ref_atac):
                #get rand shift amnt from dna
                ref_pth_fnd = [i for (i, v) in zip(ref_pth,
                                                   [os.path.isfile(x) for x in ref_pth]) if v]
                dat_dna = np.load(ref_pth_fnd[0])
                rand_seq_shift_amt = dat_dna['rand_seq_shift_amt']
                #don't bother loading dna - speed
                ref_all = data_generator.load(pos=strt_i,chro=chro,cell=cell,
                                              rand_seq_shift_amt=rand_seq_shift_amt,
                                              return_dna=False)
                ref_seq = ref_all[0]
                #save for next time
                # local chromatin accessibility signature
                np.savez(ref_atac_lcl_pth[0],chrom_access_lcl=ref_seq['chrom_access_lcl'])
                # global chromatin accessibility signature
                np.savez(ref_atac_pth[0],chrom_access_250=ref_seq['chrom_access_gbl'])
                #if using, need to load chrom access & dna
                if not no_pred:
                    ref_seq = {'dna':dat_dna['dna'],
                               'chrom_access_250':ref_seq['chrom_access_gbl'],
                               'chrom_access_lcl':ref_seq['chrom_access_lcl']}
                    
        else:
            if not no_pred:
                #get pth for it
                ref_pth_fnd = [i for (i, v) in zip(ref_pth,
                                                   [os.path.isfile(x) for x in ref_pth]) if v]
                ref_atac_pth_fnd = [i for (i, v) in zip(ref_atac_pth, 
                                                        [os.path.isfile(x) for x in ref_atac_pth]) if v]
                ref_atac_lcl_pth_fnd = [i for (i, v) in zip(ref_atac_lcl_pth, 
                                                            [os.path.isfile(x) for x in ref_atac_lcl_pth]) if v]
                dat_dna = np.load(ref_pth_fnd[0])
                dat_atac = load_gbl_atac_ref(ref_atac_pth_fnd[0])
                dat_atac_lcl = np.load(ref_atac_lcl_pth_fnd[0])
                ref_seq = {'dna':dat_dna['dna'],
                           'chrom_access_250':dat_atac['chrom_access_250'],
                           'chrom_access_lcl':dat_atac_lcl['chrom_access_lcl']}
                rand_seq_shift_amt = dat_dna['rand_seq_shift_amt']
                del dat_dna 
        #Now impute SNP for alt
        #pass same rand shift amount
        #only generate if not saved
        any_alt = any([os.path.isfile(x) for x in alt_pth])
        if (not any_alt): #or (not any_ref_atac):
            #need rand_seq_shift_amt from ref so need to load it if it isn't already
            if 'rand_seq_shift_amt' not in locals():
                #get pth for it
                ref_pth_fnd = [i for (i, v) in zip(ref_pth,
                                                   [os.path.isfile(x) for x in ref_pth]) if v]
                dat_dna = np.load(ref_pth_fnd[0])
                rand_seq_shift_amt = dat_dna['rand_seq_shift_amt']
            ref_atac_pth = [i+f'/{cell_id}_ATAC.npz' for i in checkpoint_pth]
            ref_atac_lcl_pth = [i+f'/{lcl_CA_strt_i}_ATAC_lcl.npz' for i in checkpoint_pth]
            any_ref_atac = any([os.path.isfile(x) for x in ref_atac_pth])
            any_ref_atac_lcl = any([os.path.isfile(x) for x in ref_atac_lcl_pth])    
            if (not any_ref_atac) or (not any_ref_atac_lcl):    
                print('Error: This should have been created in ref if statements, not again in alt')
                print(fail)
            if not any_alt:
                #don't bother loading chrom access - speed
                alt_all = data_generator.load(pos=strt_i,chro=chro,cell=cell,
                                              snp_pos=pos_i,snp_base=alt,
                                              rand_seq_shift_amt=rand_seq_shift_amt,
                                              return_chrom_access=False)
                alt_seq = alt_all[0]
                #save for next time
                np.savez(alt_pth[0],dna=alt_seq)
                #if using, need to load chrom access
                if not no_pred:
                    if 'dat_atac' not in locals():
                        #get pth for it
                        ref_atac_pth_fnd = [i for (i, v) in zip(ref_atac_pth, 
                                                                [os.path.isfile(x) for x in ref_atac_pth]) if v]
                        dat_atac = load_gbl_atac_ref(ref_atac_pth_fnd[0])
                    if 'dat_atac_lcl' not in locals():
                        #get pth for it
                        ref_atac_lcl_pth_fnd = [i for (i, v) in zip(ref_atac_lcl_pth, 
                                                            [os.path.isfile(x) for x in ref_atac_lcl_pth]) if v]
                        dat_atac_lcl = np.load(ref_atac_lcl_pth_fnd[0])    
                    alt_seq = {'dna':alt_seq,
                               'chrom_access_250':dat_atac['chrom_access_250'],
                               'chrom_access_lcl':dat_atac_lcl['chrom_access_lcl']}
                    del dat_atac, dat_atac_lcl
        else:
            if not no_pred:
                #get pth for it
                alt_pth_fnd = [i for (i, v) in zip(alt_pth, 
                                                   [os.path.isfile(x) for x in alt_pth]) if v]
                dat_dna = np.load(alt_pth_fnd[0])
                if 'dat_atac' not in locals():
                    #get pth for it
                    ref_atac_pth_fnd = [i for (i, v) in zip(ref_atac_pth, 
                                                            [os.path.isfile(x) for x in ref_atac_pth]) if v]
                    dat_atac = load_gbl_atac_ref(ref_atac_pth_fnd[0])
                if 'dat_atac_lcl' not in locals():
                        #get pth for it
                        ref_atac_lcl_pth_fnd = [i for (i, v) in zip(ref_atac_lcl_pth, 
                                                            [os.path.isfile(x) for x in ref_atac_lcl_pth]) if v]
                        dat_atac_lcl = np.load(ref_atac_lcl_pth_fnd[0])    
                alt_seq = {'dna':dat_dna['dna'],
                               'chrom_access_250':dat_atac['chrom_access_250'],
                               'chrom_access_lcl':dat_atac_lcl['chrom_access_lcl']}
                del dat_atac, dat_dna, dat_atac_lcl 
        if not no_pred: #don't waste time predicting if not necessary
            ref_seq_dna_all.append(ref_seq['dna'])
            ref_seq_prom_all.append(ref_seq['chrom_access_250'])
            ref_seq_lcl_all.append(ref_seq['chrom_access_lcl'])
            alt_seq_dna_all.append(alt_seq['dna'])
            alt_seq_prom_all.append(alt_seq['chrom_access_250'])
            alt_seq_lcl_all.append(alt_seq['chrom_access_lcl'])
            if ind == (len(dna_strt)-1):
                #too memory intensive to pred all at once so split
                del ref_seq,alt_seq
                num_preds = 1
                ref_seq_all_1 = {"dna":tf.concat(ref_seq_dna_all,axis=0),
                                 "chrom_access_gbl":tf.concat(ref_seq_prom_all,axis=0),
                                 "chrom_access_lcl":tf.concat(ref_seq_lcl_all,axis=0),
                                }
                del ref_seq_dna_all,ref_seq_prom_all,ref_seq_lcl_all
                alt_seq_all_1 = {"dna":tf.concat(alt_seq_dna_all,axis=0),
                                 "chrom_access_gbl":tf.concat(alt_seq_prom_all,axis=0),
                                 "chrom_access_lcl":tf.concat(alt_seq_lcl_all,axis=0)
                              }
                del alt_seq_dna_all,alt_seq_prom_all,alt_seq_lcl_all
                #Now both ref and alt are ready to be passed to model
                pred_ref = model.predict(ref_seq_all_1,return_arcsinh = False)
                pred_alt = model.predict(alt_seq_all_1,return_arcsinh = False)
                #Need to know that there needs to be a first and last i.e. one wasn't removed because 
                #too close to an edge
                max_num_pos = 1 + math.ceil((window_size_dna-target_bp)/(target_bp/2))
                #get index of centred snp pos
                centre_ind = np.where([i == window_size_dna//2 for i in snp_pos])[0][0]
                #check if upstream or downstream missing
                up_miss = False
                down_miss = False
                if len(dna_strt)<max_num_pos and centre_ind<math.ceil(max_num_pos/2)-1:
                    up_miss=True
                if len(dna_strt)<max_num_pos and centre_ind==math.ceil(max_num_pos/2)-1:
                    down_miss=True  
                #now calc for edges if present  
                #edge prediction region results may need to be chopped if overlap previous
                #this will be the case when the target bp region isn't a mutliple of the input 
                #region
                #this will be the first and last region - NOTE dna start positions must be in order
                if(dna_strt[1]-dna_strt[0]<target_bp):
                    #first so chop end
                    #remember 3 and 4 are rev comp so take from start not end
                    chop_bp = target_bp-(dna_strt[1]-dna_strt[0])
                    chop_pos = chop_bp//pred_resolution
                    pred_ref_first = np.concatenate((pred_ref[0:2,0:target_length-chop_pos,:], 
                                                     pred_ref[2:4,chop_pos:target_length,:]), 
                                                    axis=0)
                    pred_ref = pred_ref[4:,:,:]
                    pred_alt_first = np.concatenate((pred_alt[0:2,0:target_length-chop_pos,:], 
                                                     pred_alt[2:4,chop_pos:target_length,:]), 
                                                    axis=0)
                    pred_alt = pred_alt[4:,:,:]
                if(dna_strt[len(dna_strt)-1]-dna_strt[len(dna_strt)-2]<target_bp):    
                    #last so chop start
                    #remember 3 and 4 are rev comp so take from start not end
                    chop_bp = target_bp-(dna_strt[len(dna_strt)-1]-dna_strt[len(dna_strt)-2])
                    chop_pos = chop_bp//pred_resolution
                    lst_len = pred_alt.shape[0]
                    pred_ref_last = np.concatenate((pred_ref[0:2,chop_pos:target_length,:], 
                                                    pred_ref[2:4,0:target_length-chop_pos,:]), 
                                                   axis=0)
                    pred_ref = pred_ref[:lst_len-4,:,:]
                    pred_alt_last = np.concatenate((pred_alt[0:2,chop_pos:target_length,:], 
                                                    pred_alt[2:4,0:target_length-chop_pos,:]), 
                                                   axis=0)
                    pred_alt = pred_alt[:lst_len-4,:,:]
                del ref_seq_all_1,alt_seq_all_1
                
                #combine so the full window size is together
                wind_size_pred_ref = []
                wind_size_pred_alt = []
                #take every 4th channel
                n = pred_ref.shape[0]//4
                out_c = pred_ref.shape[2]
                pos_pred = pred_ref.shape[1]
                #norm
                wind_size_pred_ref.append(tf.reshape(pred_ref[::4,:,:],
                                                     (1,pos_pred*n,out_c)))
                wind_size_pred_alt.append(tf.reshape(pred_alt[::4,:,:],
                                                     (1,pos_pred*n,out_c)))
                #rand perm
                wind_size_pred_ref.append(tf.reshape(pred_ref[1:,:,:][::4,:,:],
                                                     (1,pos_pred*n,out_c)))
                wind_size_pred_alt.append(tf.reshape(pred_alt[1:,:,:][::4,:,:],
                                                     (1,pos_pred*n,out_c)))
                #rev comp
                wind_size_pred_ref.append(tf.reshape(pred_ref[2:,:,:][::4,:,:],
                                                     (1,pos_pred*n,out_c)))
                wind_size_pred_alt.append(tf.reshape(pred_alt[2:,:,:][::4,:,:],
                                                     (1,pos_pred*n,out_c)))
                #rev comp + rand perm
                wind_size_pred_ref.append(tf.reshape(pred_ref[3:,:,:][::4,:,:],
                                                     (1,pos_pred*n,out_c)))
                wind_size_pred_alt.append(tf.reshape(pred_alt[3:,:,:][::4,:,:],
                                                     (1,pos_pred*n,out_c)))
    if no_pred:#don't waste time predicting if not necessary
        return(None)
    #combine to 4 dim for each
    wind_size_pred_ref = tf.concat(wind_size_pred_ref,axis=0)
    wind_size_pred_alt = tf.concat(wind_size_pred_alt,axis=0)
    #first check if first and last modified and append first
    if(dna_strt[1]-dna_strt[0]<target_bp and not up_miss):
        wind_size_pred_ref = tf.concat([pred_ref_first,wind_size_pred_ref],axis=1)
        wind_size_pred_alt = tf.concat([pred_alt_first,wind_size_pred_alt],axis=1)
    if(dna_strt[len(dna_strt)-1]-dna_strt[len(dna_strt)-2]<target_bp and not down_miss):
        wind_size_pred_ref = tf.concat([wind_size_pred_ref,pred_ref_last],axis=1)
        wind_size_pred_alt = tf.concat([wind_size_pred_alt,pred_alt_last],axis=1)
    #Now aggregate the SNP effect based on chosen method
    #loop for each of org, rand perm, rev comp and average
    if effect_mode=='both':
        eff_max = []
        eff_sum = []
        for i in range(wind_size_pred_ref.shape[0]):
            max_i,sum_i = effect_func(wind_size_pred_ref[i,:,:],wind_size_pred_alt[i,:,:])
            eff_max.append(max_i)
            eff_sum.append(sum_i)
        agg_eff_max = np.mean(np.vstack(eff_max),axis=0)    
        agg_eff_sum = np.mean(np.vstack(eff_sum),axis=0)
        return(agg_eff_max,agg_eff_sum)
    else:    
        eff=[]        
        for i in range(wind_size_pred_ref.shape[0]):
            eff.append(effect_func(wind_size_pred_ref[i,:,:],wind_size_pred_alt[i,:,:]))
        agg_eff = np.mean(np.vstack(eff),axis=0)
        return(agg_eff)
    
    
     
