import tensorflow as tf
import numpy as np
import os
from EnformerCelltyping.utils import(gelu,
                                     create_enf_chopped_model,
                                     pearsonR)

def build_enf_celltyping(use_prebuilt_model: bool = True,
                         enf_celltyping_pth: str = None,
                         window_size_CA: int= 1562*128, 
                         dropout_rate: float = 0.4,
                         n_128bp_factors: int = 25,
                         n_500bp_factors: int = 40,
                         n_5kbp_factors: int = 45,
                         n_gbl_factors: int =25,     
                         ct_embed_nodes: int = 3072,
                         dna_nodes: int = 1536,
                         n_dense_dna: int = 1,
                         n_dense_ct: int = 2,
                         pointwise_nodes_dna: int = 3072,
                         output_activation_dna: str ="softplus",
                         output_activation_ct: str ="linear",
                         output_channels: int = 6
                         ):
    """
    Use this to build the Enformer Celltyping model either with a
    new architecture or by loading a previous model.
    
    Arguments:
    
    use_prebuilt_model: 
        Whether to load a model or create a new architecture.
        The default is True.
    
    enf_celltyping_pth: 
        Path to the weights/whole saved model of enformer celltyping 
        model if use_prebuilt_model is True.
    
    window_size_CA: 
        The number of base-pair positions used to embed the cell
        type representations by chromatin accessibility. The default is 
        1562*128 bp.
    
    dropout_rate: 
        The level of dropout used throughout architecture. The
        default is 0.4.
    
    n_128bp_factors:
        The number of factors used to represent the genome local cell 
        type embedding at 128bp resolution. Default is 25.
    
    n_500bp_factors: 
        The number of factors used to represent the genome local cell 
        type embedding at 250bp resolution. Default is 40.
    
    n_5kbp_factors: 
        The number of factors used to represent the genome local cell 
        type embedding at 5,000bp resolution. Default is 45.
    
    n_gbl_factors:    
        The number of factors used to represent the genome global cell 
        type embedding at 250bp resolution. Default is 25.
        
    ct_embed_nodes:
        The size of dense layer used for cell type embedding. The
        default is 3072.
        
    dna_nodes:
        The size of the dense layer used for DNA embedding. Default is 
        1536.
        
    n_dense_dna:
        The number of dense layer blocks to use for the dna channel. 
        The default is 1.    
        
    n_dense_ct:
        The number of dense layer blocks to use for the chromatin 
        accessibility data after embedding. Default is 2.
        
    pointwise_nodes_dna:
        The output number of nodes for the pointwise convolution for the
        DNA channel. If less than input size, dimensionality reduction. 
        Default is 3072.
        
    output_activation_dna:
        The activation to use for the final output dense layer for the DNA
        channel. Default is softplus.
    
    output_activation_ct:
        The activation to use for the final output dense layer for the cell
        typing channel. Default is linear.
    
    output_channels:
        The number of output channels (e.g. histone marks) to predict by 
        the model. The default is 6 corresponding to the 6 histone marks
        Enformer Celltyping was trained on: 
        ['h3k27ac', 'h3k4me1', 'h3k27me3', 'h3k4me3', 'h3k36me3' and 'h3k9me3']
    
    ENF_CHANNELS,ENF_PRED_POS:
        output shape of chopped Enformer model
    GBL_EMBED_BP:
        size in base-pairs of the global chromatin acessibility embedding
    """
    #check if just weights or whole model saved
    if use_prebuilt_model:
        assert enf_celltyping_pth is not None, ('Give path to enformer tf.hub model')
        filename, file_extension = os.path.splitext(enf_celltyping_pth)
        if file_extension != '.h5':
            return(tf.keras.models.load_model(enf_celltyping_pth,
                                              custom_objects={'pearsonR': pearsonR}))
    #else define new model architecture
    inputs = dict(
        dna= tf.keras.layers.Input(
            shape=(ENF_PRED_POS, ENF_CHANNELS),
            name='dnaInput'
        ),
        #local chrom access embed
        chrom_access_lcl= tf.keras.layers.Input(
            shape=(window_size_CA//128,),
            name='ChromAccessLclInput'
        ),
        #global chrom access embed
        chrom_access_gbl= tf.keras.layers.Input(
            shape=(GBL_EMBED_BP//250,),
            name='ChromAccessGblInput'
        )
    )
    #------------------------------------
    #Chrom Access - Cell type info arch
    #need positive inputs for embedding - set min, max cut-off change of arcsinh -log10 p-val of 5
    #analysis of max values in data in max_-log10pval_signal_train_cells.ipynb but in short <5.
    max_abs_diff = 5
    norm_chrom_access = tf.add(tf.clip_by_value(inputs['chrom_access_lcl'],
                                                clip_value_min=-max_abs_diff, 
                                                clip_value_max=max_abs_diff),
                               max_abs_diff)
    #Embedding - get embedded factors and flatten
    embed128 = tf.keras.layers.Embedding(window_size_CA//128,
                                         n_128bp_factors,
                                         input_length=1,
                                         name="128bp_embedding")(norm_chrom_access)
    embed500 = tf.keras.layers.Embedding((window_size_CA//(128*4))+1,
                                         n_500bp_factors,
                                         input_length=1,
                                         name="500bp_embedding")(norm_chrom_access)
    embed5k = tf.keras.layers.Embedding((window_size_CA//(128*40))+1,
                                        n_5kbp_factors,
                                        input_length=1,
                                        name="5kbp_embedding")(norm_chrom_access)
    
    #Embed global
    embedGbl = tf.keras.layers.Embedding(GBL_EMBED_BP//250,
                                         n_gbl_factors,
                                         input_length=1,
                                         name="gbl_embedding")(inputs['chrom_access_gbl'])
    
    #flatten
    embed128 = tf.keras.layers.Flatten(name="flatten128")(embed128)
    embed500 = tf.keras.layers.Flatten(name="flatten500")(embed500)    
    embed5k = tf.keras.layers.Flatten(name="flatten5k")(embed5k)
    embedGbl = tf.keras.layers.Flatten(name="flattenGbl")(embedGbl)
    #combine        
    cat_embed = tf.keras.layers.Concatenate(
                                            name="cat_embed")([embed128,
                                                               embed500,
                                                               embed5k,
                                                               embedGbl])
    #2 dense layers - mimic Avocado - keep input shape the same
    for i in range(1,n_dense_ct+1):
        cat_embed = tf.keras.layers.Dense(ct_embed_nodes, activation=gelu, 
                                          name=f"dense_combn{i}_celltype")(cat_embed)
        #dropout
        cat_embed = tf.keras.layers.Dropout(dropout_rate/4,#/8,#4,
                                           name=f"dropout{i}_celltype")(cat_embed)
    
    all_chan = []
    for i in range(1,output_channels+1):
            cat_embed2 = tf.keras.layers.Dense(ENF_PRED_POS, activation=output_activation_ct,
                                               name=f"dense_chan_celltype_{i}")(cat_embed)
            cat_embed2 = tf.keras.layers.Dropout(dropout_rate/8,#4,
                                                 name=f"dropout_chan_celltype_{i}")(cat_embed2)
            #expand dim add dim at end
            cat_embed2 = tf.expand_dims(cat_embed2, axis=-1)
            #append channel
            all_chan.append(cat_embed2)
    output_delta = tf.keras.layers.concatenate(all_chan, axis=2,name="delta")
    
    #------------------------------------

    #------------------------------------
    #DNA arch
    x = inputs['dna']
    for i in range(1,n_dense_dna+1):
        #Dense layer (connections on last dimension) so celltype embedding and DNA joined
        x = tf.keras.layers.Dense(dna_nodes, activation=gelu, 
                                  name=f"dense{i}_dna")(x)
        #dropout
        x = tf.keras.layers.Dropout(dropout_rate/16,
                                    name=f"dropout{i}_dna")(x)
    #final_pointwise-----
    #conv block---
    x = tf.keras.layers.BatchNormalization(
        momentum=0.99,epsilon=0.001,
        center=True,scale=True,
        name="batchnorm1_finalpoint")(x)
    #gelu - https://arxiv.org/abs/1606.08415
    x = gelu(x,name="gelu1_convblck_finalpoint")
    #conv
    x = tf.keras.layers.Conv1D(filters=pointwise_nodes_dna,kernel_size=1,
                               name="conv1d1_finalpoint")(x)
    #---
    #dropout
    x = tf.keras.layers.Dropout(dropout_rate/16,
                                name="dropout1_finalpoint")(x)
    #gelu - https://arxiv.org/abs/1606.08415
    x = gelu(x,name="gelu1_finalpoint")
    #-----
    #Dense layer for output
    output_avg = tf.keras.layers.Dense(output_channels,
                                        activation=output_activation_dna,name="avg")(x)
    model = tf.keras.Model(inputs, [output_avg,output_delta], name = "EnfCelltyping")
    #load wieghts if necessary
    if use_prebuilt_model and file_extension == '.h5':
        model.load_weights(enf_celltyping_pth)
    return(model)


#set model values ---------
#DNA input same as Enformer - DNA input window, ~100kbp either side
WINDOW_SIZE_DNA = 196_608
#output from chopped enformer model
ENF_CHANNELS = 1536
ENF_PRED_POS = 896
GBL_EMBED_BP = 1216*3_000 #PanglaoDB 1216 marker genes, 3k around TSS
#---------

class Enformer_Celltyping(object):
    """
    Enformer Celltyping is an extension of the enformer mdoel first
    proposed by Avsec et al., 2021. Enformer Celltyping was originally 
    trained to predict 6 histone marks (h3k27ac, h3k4me1, h3k27me3, 
    h3k4me3, h3k36me3 and h3k9me3) in previously unseen cell types. 
    The model can generalise to new cell types by using chromatin 
    accessibility data (ATAC-Seq). The chromatin accessibility data is 
    used to embed a representation of a cell type which is used to adjust
    the peaks derived from DNA using Enformer. The chromatin accessibility 
    data is embedded at 250bp and 5kbp resolution, mimicing the genomic 
    embedding approach used by Avocado (Schreiber et al.,2020).
    
    Lift weights from pre-trained Enformer model available in tf.hub
    to use as a pre-trained model by Enformer Celltyping. The attention
    mechanism in Enformer enables the prediction of long range interactions
    of DNA (up to 100Kbp around the predicted region). The custom
    class model can be used to load the Tnsorflow Keras Enformer Celltyping 
    model or create a new model arcitecture. The model can then be used to 
    predict in new cell types. 
    
    Arguments:
    
    assays:
        assays Enformer celltyping will predict. It was trained on 
        ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3']
          
    enf_path:
        path to the saved tf.hub Enformer model from which the weights
        will be extracted to use as a pre-trained model. Default is None
        which will not apply a transformation with the Enformer Chopped model.
    
    use_prebuilt_model: 
        Whether to load a model or create a new architecture.
        The default is True.
    
    enf_celltyping_pth: 
        Path to the weights of enformer celltyping model if 
        use_prebuilt_model is True.
    
    window_size_CA: 
        The number of base-pair positions used to embed the cell
        type representations by chromatin accessibility. The default is 
        1562*128 bp.
    
    dropout_rate: 
        The level of dropout used throughout architecture. The
        default is 0.4.
        
    n_128bp_factors:
        The number of factors used to represent the genome cell type
        embedding at 128bp resolution. Default is 25.
    
    n_500bp_factors: 
        The number of factors used to represent the genome cell type
        embedding at 250bp resolution. Default is 40.
    
    n_5kbp_factors: 
        The number of factors used to represent the genome cell type 
        embedding at 5,000bp resolution. Default is 45.
        
    n_gbl_factors:    
        The number of factors used to represent the genome global cell 
        type embedding at 250bp resolution. Default is 25.    
    
    ct_embed_nodes:
        The size of dense layer used for cell type embedding. The
        default is 3072.
        
    dna_nodes:
        The size of the dense layer used for DNA embedding. Default is 
        1536.
    
    n_dense_dna:
        The number of dense layer blocks to use for the dna channel. 
        The default is 1.
    
    n_dense_ct:
        The number of dense layer blocks to use for the chromatin 
        accessibility data after embedding. Default is 2.     
    
    pointwise_nodes_dna:
        The output number of nodes for the pointwise convolution for the
        DNA channel. If less than input size, dimensionality reduction. 
        Default is 3072.
        
    output_activation_dna:
        The activation to use for the final output dense layer for the DNA
        channel. Default is softplus.
    
    output_activation_ct:
        The activation to use for the final output dense layer for the cell
        typing channel. Default is linear.
    
    Returns Enformer Celltyping class object
    """
    def __init__(self,  
                 assays: list = ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3'],
                 enf_path = None, 
                 use_prebuilt_model=True,
                 enf_celltyping_pth: str = None,
                 window_size_CA: int= 1562*128,
                 dropout_rate: float = 0.4,
                 n_128bp_factors: int = 25,
                 n_500bp_factors: int = 40,
                 n_5kbp_factors: int = 45,
                 n_gbl_factors: int = 25,
                 ct_embed_nodes: int = 3072,
                 dna_nodes: int = 1536,
                 n_dense_ct: int = 2,
                 n_dense_dna: int = 1,
                 pointwise_nodes_dna: int = 3072,
                 output_activation_dna: str ="softplus",
                 output_activation_ct: str ="linear"):
    
        self.assays = list(assays)
        self.n_assays = len(assays)

        self.use_prebuilt_model = use_prebuilt_model
        self.enf_celltyping_pth = enf_celltyping_pth
        self.window_size_CA = window_size_CA
        self.dropout_rate = dropout_rate
        self.n_128bp_factors = n_128bp_factors
        self.n_500bp_factors = n_500bp_factors
        self.n_5kbp_factors = n_5kbp_factors
        self.n_gbl_factors = n_gbl_factors
        self.ct_embed_nodes = ct_embed_nodes
        self.n_dense_ct = n_dense_ct
        self.dna_nodes = dna_nodes
        self.n_dense_dna = n_dense_dna
        self.pointwise_nodes_dna = pointwise_nodes_dna
        self.output_activation_dna = output_activation_dna
        self.output_activation_ct = output_activation_ct
        
        if enf_path is not None:
            self.enf = create_enf_chopped_model(enf_path)
        else:
            self.enf = None

        self.model = build_enf_celltyping(use_prebuilt_model=self.use_prebuilt_model,
                                          enf_celltyping_pth=self.enf_celltyping_pth,
                                          window_size_CA=self.window_size_CA,
                                          dropout_rate=self.dropout_rate,
                                          n_128bp_factors = self.n_128bp_factors,
                                          n_500bp_factors = self.n_500bp_factors,
                                          n_5kbp_factors = self.n_5kbp_factors,
                                          n_gbl_factors = self.n_gbl_factors,
                                          ct_embed_nodes=self.ct_embed_nodes,
                                          n_dense_ct = self.n_dense_ct,
                                          dna_nodes=self.dna_nodes,
                                          n_dense_dna=self.n_dense_dna,
                                          pointwise_nodes_dna=self.pointwise_nodes_dna,
                                          output_activation_dna=self.output_activation_dna,
                                          output_activation_ct=self.output_activation_ct,
                                          output_channels=self.n_assays)
    
    def compile(self,**kwargs):
        """
        Just wrapper method for the keras compile method.
        """
        self.model.compile(**kwargs)
        
    def summary(self):
        """
        Just wrapper method for the keras summary method.
        """
        self.model.summary()
    
    def layers(self):
        """
        Just wrapper method for keras layers method.
        """
        return(self.model.layers)
        
    def trainable_weights(self):
        """
        Just wrapper method for keras trainable_weights method.
        """
        return(self.model.trainable_weights)
    
    def save(self,pth,**kwargs):
        """
        Just wrapper method for the keras save method.
        """
        self.model.save(pth,**kwargs)
    
    def input(self):
        """
        Just wrapper method for the keras model.input.
        """
        return(self.model.input)

    def create_map_function(self):
        """
        Create new data loaders wrapping current.
        Passes DNA to pre-trained Enformer Chopped model
        """
        def trans_dataloader(x,y):
            #pass dna through chopped Enformer to get Attention info
            x['dna'] = self.enf.predict_on_batch(x['dna'])
            return(x,y)
        return trans_dataloader
    
    
    def load_embedding(self, layer_name):
        """
        Returns the index of the learned embedding for 
        specified (by name) layer.
        Adapted from Avocado
        
        layer_name: 
            name of embedded layer
        Returns index of layer
        """
        
        for ind, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return layer.get_weights()

        raise ValueError(f"No layer in model named '{layer_name}'.")
    
    def fit(self,train_data,validation_data=None,
            epochs = 100,steps_per_epoch=1,
            verbose=2,callbacks=None, **kwargs):
        """
        fit does two things:
        
        1. Passes DNA input through Enformer model
        2. Passes Enformer DNA and Cell type embedding to Tensorflow Keras fit()
        
        Arguments:
    
        train_dataloader:
            Data generator for training data. Should produce tensors with:
            X = dict(
                    dna= tf.keras.layers.Input(
                            shape=(batch_size,
                                   WINDOW_SIZE_DNA, 4),
                            dtype=tf.float32, #enformer works with float32 not 64
                    ),
                    #local chrom access embed
                    chrom_access_lcl= tf.keras.layers.Input(
                            shape=(batch_size,
                                   window_size_CA//128,),
                            dtype=tf.float32,
                    ),
                    #global chrom access embed
                    chrom_access_gbl= tf.keras.layers.Input(
                            shape=(batch_size,
                                   GBL_EMBED_BP//250,),
                            dtype=tf.float32,       
                    )
                )    
            Y = dict(
                    #model preds avg and then delta based on cell type separately
                    avg = tf.TensorSpec(
                            shape=(batch_size,
                                   896,
                                   self.n_assays),
                            dtype=tf.float32, #enformer works with float32 not 64
                    ),
                    delta = tf.TensorSpec(
                            shape=(batch_size,
                                   896,
                                   self.n_assays),
                            dtype=tf.float32, #enformer works with float32 not 64
                    )
                )    
          
          valid_dataloader:
              Data generator for validation dataset (optional). Same shape as
              train_dataloader.
          
          n_epochs:
              Number of epochs to train the model
              
          steps_per_epoch:
              Number of steps to train the full epoch size.
          
          verbose:
              Verborse parameter for tensorflow keras fit()
          
          callbacks:
              Callbacks parameter for tensorflow keras fit()
        """
        if self.enf is not None:
            #set up steps to pass DNA through Enformer chopped
            trans_dataloader = self.create_map_function()
            train_data = train_data.map(trans_dataloader)
            train_data = train_data.prefetch(tf.data.AUTOTUNE)
        
            if validation_data is not None:
                validation_data = validation_data.map(trans_dataloader)
                validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

        self.model.fit(train_data,
                       epochs=epochs,
                       verbose=verbose,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=validation_data,
                       callbacks=callbacks,
                       **kwargs)
        
    def evaluate(self, X, y,**kwargs):
        """
        evaluate does two things:
        
        1. Passes DNA input through Enformer model
        2. Passes Enformer DNA and Cell type embedding to Tensorflow Keras evaluate()
        
        """
        if self.enf is not None:
            #set up steps to pass DNA through Enformer chopped
            X['dna'] = self.enf.predict_on_batch(X['dna'])
         
        output = self.model.evaluate(X,y,**kwargs)
        
        return(output)
    
    def predict(self, X, return_arcsinh: bool = False, 
                return_delta: bool = False, **kwargs):
        """
        predict does two things:
        
        1. Passes DNA input through Enformer model
        2. Passes Enformer DNA and Cell type embedding to Tensorflow Keras predict()
        
        Arguments:
        
        return_arcsinh:
            Should predictions be returned with arcsinh transformation (general approach
            is to train Enformer Celltyping on this data but when actually using the data
            for downstream work, the output should be converted back). By default this
            will be converted back.
        return_delta:
            This will return just the delta of the cell type from the average
            cell type prediction at the genomic location. Default is False which will return
            the predicted cell type signal at the location.
        """
        
        if self.enf is not None:
            #set up steps to pass DNA through Enformer chopped
            X['dna'] = self.enf.predict_on_batch(X['dna'])
         
        
        pred = self.model.predict(X,**kwargs)
        #convert avg pred and diff to one, cell type pred
        output = pred[0]+pred[1]
        #if only want delta
        if return_delta:
            output = pred[1]
        if not return_arcsinh:
            #tranform predictions back from arc-sinh back with sinh
            output = np.sinh(output)
        
        return(output)