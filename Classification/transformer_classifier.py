from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Flatten
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import Sequential
from transformer_block import TransformerBlock
from token_and_position_embedding import TokenAndPositionEmbedding

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('CPU')


# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class TransformerClassifier:

    def __init__(self, input_shape, vocabulary_size, config):
        self.set_config(config)

        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        #opt = tf.keras.optimizers.SGD()

        vocabulary_size = vocabulary_size

        # Dimension of the embedding vectors
        # latent_dim = 512
        # # Number of head in multi attention layer
        # num_heads = 8
        # # Number of predicted category
        # num_classes = 3
        # # Dimension of the feed forward sequential layer inside the transformer block
        # ff_dim = 128

        # Dimension of the embedding vectors
        latent_dim = 256
        # Number of head in multi attention layer
        num_heads = 8
        # Number of predicted category
        num_classes = self.num_classes
        # Dimension of the feed forward sequential layer inside the transformer block
        ff_dim = 64  # Hidden layer size in feed forward network inside transformer

        inputs = layers.Input(shape=(input_shape,))
        embedding_layer = TokenAndPositionEmbedding(input_shape, vocab_size=vocabulary_size + 1, embed_dim=latent_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(latent_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        # Creating and compiling the model
        self.loss = "categorical_crossentropy"
        #self.loss = "binary_crossentropy"
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(opt, loss=self.loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name='cat_acc')])

        self.checkpoint_path = os.path.join('models', 'attention_nes', 'cp-{epoch:04d}.ckpt')
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)

    def set_config(self, config):
        # self.nneurons = config['neurons']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.train_size = config['train_size']
        self.num_classes = config['num_classes']

    def fit(self, x_train, x_val, y_train, y_val, nfold):
        # cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
        #                                               save_weights_only=True,
        #                                               verbose=1,
        #                                               monitor='val_cat_acc',
        #                                               mode='max',
        #                                               save_best_only=True
        #                                               )

        early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

        initial_learning_rate = 0.0001
        final_learning_rate = 0.000001
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.epochs)
        steps_per_epoch = int(self.train_size / self.batch_size)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=False)

        lr = LearningRateScheduler(lr_schedule)

        # self.model.save_weights(self.checkpoint_path.format(epoch=0))

        history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=[early_cb, lr]
                                 #callbacks=[early_cb]
                                 )

        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.savefig(str(nfold) + '_fold_loss_graph.png')
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        plt.plot(history.history['cat_acc'], label='Training')
        plt.plot(history.history['val_cat_acc'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.savefig(str(nfold) + '_fold_accuracy_graph.png')
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        self.model.save(os.path.join('models', 'attention_nes', 'saved_model', 'fold_n', str(nfold)))
        print("Model Saved")
