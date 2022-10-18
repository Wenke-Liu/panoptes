import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from datetime import datetime


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class PANOPTES:
    def __init__(self,
                 base_model_name='InceptionResNetV1',
                 input_dim=(299, 299, 3),
                 pooling='avg',
                 dropout=0.5,
                 n_classes=2,
                 contrastive=False,
                 saved_model=None):

        self.base_model_name = base_model_name
        self.input_dim = input_dim
        self.pooling = pooling
        self.dropout = dropout
        self.n_classes = n_classes
        self.encoder_history = None
        self.classifier_history = None
        self.contrastive = contrastive
        self.model = self.build()  # build the panoptes branches

        if saved_model:  # load weights from saved model if provided
            if self.contrastive:
                self.add_classifier()

            self.model.load_weights(saved_model)
            print('Loading saved model: ' + saved_model)

        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

    def build(self):    # build the panoptes architecture, with options of softmax/acitvation outputs
        input_a = keras.Input(shape=self.input_dim, name='input_a')
        input_b = keras.Input(shape=self.input_dim, name='input_b')
        input_c = keras.Input(shape=self.input_dim, name='input_c')
        
        if self.base_model_name == 'InceptionResNetV1':
            keras_app = __import__('layer',
                                   fromlist=[self.base_model_name])  # customized InceptionResNetV1
        else:
            keras_app = __import__('tensorflow.keras.applications',  # import a keras base model
                                   fromlist=[self.base_model_name])

        print('Using base model: ' + self.base_model_name)
        base_model = getattr(keras_app, self.base_model_name)

        branch_a = base_model(weights=None, include_top=False, pooling=self.pooling, input_tensor=input_a)
        branch_a._name = self.base_model_name + '_a'
        for layer in branch_a.layers:
            layer._name = 'branch_a_' + layer.name

        branch_b = base_model(weights=None, include_top=False, pooling=self.pooling, input_tensor=input_b)
        branch_b._name = self.base_model_name + '_b'
        for layer in branch_b.layers:
            layer._name = 'branch_b_' + layer.name

        branch_c = base_model(weights=None, include_top=False, pooling=self.pooling, input_tensor=input_c)
        branch_c._name = self.base_model_name + '_c'
        for layer in branch_c.layers:
            layer._name = 'branch_c_' + layer.name

        xa = branch_a(input_a)
        xb = branch_b(input_b)
        xc = branch_c(input_c)

        x = keras.layers.Concatenate(axis=-1, name='activation')([xa, xb, xc])  # activation layer
        x = keras.layers.Dropout(self.dropout, name='activation_dropout')(x)

        if self.contrastive:    # contrastive learning, first stage model built, output the activations
            print('Contrastive learing. No classifier layers. Projection head added.')
            out = tf.keras.layers.Dense(128, name='projection', activation='relu')(x)
        
        else:
            out = tf.keras.layers.Dense(self.n_classes, name='prob', activation='softmax')(x)  # prob. output


        panoptes_model = keras.Model(inputs=[input_a, input_b, input_c],
                                     outputs=out, name='panoptes')
        
        print(panoptes_model.summary())

        print('Activation layer size: ' + str(panoptes_model.get_layer('activation').output.shape[1]))

        return panoptes_model

    def add_classifier(self, trainable=False):    # for contrastive learning, add classifier after pretraining.
        print('Adding classifer.')
        print(self.model.trainable)
        encoder = self.model
        encoder.trainable = trainable
        for layer in encoder.layers:
            layer.trainable = trainable
        print(self.model.trainable)
        print(self.model.summary())
        inputs = encoder.input
        x = encoder.get_layer('activation_dropout').output
        out = tf.keras.layers.Dense(self.n_classes, name='prob', activation='softmax')(x)
        self.model = keras.Model(inputs=inputs, outputs=out)
        print(self.model.trainable)
        print(self.model.summary())

    def compile(self,
                loss_fn=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC()],
                learning_rate=0.0001):

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=loss_fn,
                           metrics=metrics
                           )

    def train(self,
              trn_data, val_data,
              classifier_loss=tf.keras.losses.CategoricalCrossentropy(),
              contrastive_temp=None,
              n_epoch=10,
              steps=10000,
              patience=5,
              log_dir='./log',
              model_dir='./model',
              class_weight=None,
              ):
        
        os.makedirs(log_dir, exist_ok=True)
        print('Training logs in: ' + log_dir)
        
        os.makedirs(model_dir, exist_ok=True)
        print('Saving model in: ' + model_dir)

        os.makedirs(model_dir + '/ckpt', exist_ok=True)

        csv_logger = tf.keras.callbacks.CSVLogger(log_dir + '/trn_history_logs.csv', append=True)

        tensor_board = tf.keras.callbacks.TensorBoard(log_dir + '/trn_tb_logs', update_freq=1000)
         
        ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + '/ckpt/weights.{epoch:03d}-{val_loss:.4f}.hdf5',
                                                  save_weights_only=True,
                                                  monitor='val_loss',
                                                  mode='min',
                                                  save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=patience,
                                                          restore_best_weights=True)

        if not self.contrastive:     # one stage training
            self.compile(loss_fn=classifier_loss,
                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            self.classifier_history = self.model.fit(trn_data, validation_data=val_data,    # train step
                                                     steps_per_epoch=steps,
                                                     epochs=n_epoch,
                                                     callbacks=[csv_logger, tensor_board, ckpt, early_stopping])
        
        else:
            self.compile(learning_rate=0.0001, 
                         loss_fn=SupervisedContrastiveLoss(contrastive_temp),
                         metrics=[])
            
            print('Contrastive learning: starts pretraining...')

            pre_csv_logger = tf.keras.callbacks.CSVLogger(log_dir + '/pre_trn_history_logs.csv', append=True)

            pre_tensor_board = tf.keras.callbacks.TensorBoard(log_dir + '/pre_trn_tb_logs', update_freq=1000)
         
            pre_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + '/ckpt/pre-weights.{epoch:03d}-{val_loss:.4f}.hdf5',
                                                          save_weights_only=True,
                                                          monitor='val_loss',
                                                          mode='min',
                                                          save_best_only=True)

            self.encoder_history = self.model.fit(trn_data, validation_data=val_data,
                                                  steps_per_epoch=steps,
                                                  epochs=n_epoch,
                                                  callbacks=[pre_csv_logger, pre_tensor_board, pre_ckpt, early_stopping])
            
            
            self.add_classifier()
            self.compile(loss_fn=classifier_loss,    # compile again for classifier training
                         metrics=tf.keras.metrics.SparseCategoricalAccuracy())    
            print('Training classifier...')
            self.classifier_history = self.model.fit(trn_data, validation_data=val_data,
                                                    steps_per_epoch=steps,
                                                    epochs=n_epoch,
                                                    class_weight=class_weight,
                                                    callbacks=[csv_logger, tensor_board, ckpt, early_stopping])
            

        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
        
        self.model.save_weights(model_dir + "/panoptes_weights_final.h5")
        print('Final model saved.')

    def inference(self, x, batch_size=8):
        inputs = self.model.input
        activation = self.model.get_layer('activation').output
        prob = self.model.get_layer('prob').output
        inference_model = keras.Model(inputs=inputs,
                                      outputs=[activation, prob])
        if isinstance(x, np.ndarray):    # if input is numpy array
            res = inference_model.predict(x, batch_size=batch_size)
        else:    # if input is tf dataset
            res = inference_model.predict(x)
        return res

    def print_attr(self):
        print('Model attributes:')
        print('Base model name: ' + str(self.base_model_name))
        print('Contrastive learning: ' + str(self.contrastive))
        print('Input size: ' + str(self.input_dim))
        print('Pooling function: ' + str(self.pooling))
        print('Dropout: ' + str(self.dropout))
        print('Number of outcome classes: ' + str(self.n_classes))
        print('Model date-time: ' + str(self.datetime))


