def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class mit():
    def __init__(self):
        super(mit, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['N', 'S', 'V', 'F', 'Q']
        self.sequence_len = 186

        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128

        # Transformer original (mantenido para compatibilidad)
        self.trans_dim = 25
        self.num_heads = 5
        
        # Parámetros para Variable Temporal Attention (OPTIMIZADOS para 6GB GPU)
        self.embedding_dims = 128  # Reducido de 256 a 128
        self.num_transformer_blocks = 2  # Reducido de 3 a 2
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        self.time_emb = 4


class ptb():
    def __init__(self):
        super(ptb, self).__init__()
        # data parameters
        self.num_classes = 2
        self.class_names = ['normal', 'abnormal']
        self.sequence_len = 188

        # model configs
        self.input_channels = 1  # 15
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128

        # Transformer original (mantenido para compatibilidad)
        self.trans_dim = 25
        self.num_heads = 5
        
        # Parámetros para Variable Temporal Attention (OPTIMIZADOS para 6GB GPU)
        self.embedding_dims = 128  # Reducido de 256 a 128
        self.num_transformer_blocks = 2  # Reducido de 3 a 2
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        self.time_emb = 4