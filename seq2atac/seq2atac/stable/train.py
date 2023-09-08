from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
def train_classifier(model,train_gen,val_gen,save_path):
    print("Enabling callbacks...")
    early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience = 3, restore_best_weights = True)
    ## model checkpoint
    checkpoint_callback = ModelCheckpoint(save_path,
                                        monitor='val_accuracy', 
                                        mode='max', 
                                        verbose=1, 
                                        save_best_only=True,
                                        save_weights_only=True)

    ## FIT
    print("Fitting...")
    model.fit(x=train_gen,
            epochs=15,
            verbose=1,
            callbacks=[early_stopping_callback,checkpoint_callback,History()],
            validation_data=val_gen,
            class_weight=None,
            initial_epoch=0,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen),
            workers=1
        )  
    return model

    