from seq2atac.stable.models.convolutional import get_bpnet_model, get_bpnet_phylop_seq_model
from seq2atac.stable.models.transformers import get_tx_model
from seq2atac.stable.models.convolutional_phylop import get_bpnet_phylop_seq_model_big, get_motifnet

model_name_to_fn = {"conv_1364": lambda : get_bpnet_model(1364,8),
                    "conv_phylop_seq2seq_1364": lambda: get_bpnet_phylop_seq_model(1364,8),
                    "conv_phylop_seq2seq_big_1364": lambda: get_bpnet_phylop_seq_model_big(1364,8),
                    "motifnet": get_motifnet,
                    "tx_1364": lambda: get_tx_model(1364,8,2)}