from seq2atac.stable.models.convolutional import get_bpnet_model

model_name_to_fn = {"conv_1364": lambda : get_bpnet_model(1364,8)}