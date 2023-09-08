import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add, Input, Conv1D, BatchNormalization, GlobalAvgPool1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Activation, Flatten, Lambda, Cropping1D
# from keras.optimizers import Adam
# from tensorflow.keras.backend import int_shape

def bpnet_model(inp,filters,conv1_kernel_size,n_dil_layers,dil_kernel_size):
    
    first_conv = Conv1D(filters,
                        kernel_size=conv1_kernel_size,
                        padding='valid', 
                        activation='relu',
                        name='1st_conv')(inp)
    res_layers = [(first_conv, '1stconv')]
    layer_names = [str(i) for i in range(1,n_dil_layers+1)]
    
    for i in range(1, n_dil_layers + 1):
        if i == 1:
            res_layers_sum = first_conv
        else:
            res_layers_sum = Add(name='add_{}'.format(i))([l for l, _ in res_layers])
        conv_layer_name = '{}conv'.format(layer_names[i-1])
        conv_output = Conv1D(filters, 
                             kernel_size=3, 
                             padding='valid',
                             activation='relu', 
                             dilation_rate=2**i,
                             name=conv_layer_name)(res_layers_sum)
        conv_output_shape = conv_output.shape.as_list()
        cropped_layers = []
        for lyr, name in res_layers:
            lyr_shape = lyr.shape.as_list()
            cropsize = int(lyr_shape[1]/2) - int(conv_output_shape[1]/2)
            lyr_name = '{}-crop_{}th_dconv'.format(name.split('-')[0], i)
            cropped_layers.append((Cropping1D(cropsize,
                                              name=lyr_name)(lyr),
                                  lyr_name))
        cropped_layers.append((conv_output, conv_layer_name))
        res_layers = cropped_layers

    combined_conv = Add(name='combined_conv')([l for l, _ in res_layers])
    return combined_conv


def get_bpnet_model(input_width, num_blocks):
    inp = Input(shape=(input_width, 4),name='sequence')
    out = bpnet_model(inp,128,21,num_blocks,3)
    out = Flatten(name="flatten")(out)
    out = Dense(16,activation="relu",name="dense_1")(out)
    out = BatchNormalization(name="bn_d1")(out)
    out = Dense(8,activation="relu",name="dense_0")(out)
    out = BatchNormalization(name="bn_d0")(out)
    out = Dense(1,activation="sigmoid",name="dnase")(out)

    model=Model(inputs=[inp],outputs=[out])
    model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
    #model.summary()
    return model

