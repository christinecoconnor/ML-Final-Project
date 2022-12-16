import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

"""
    Useful constants for later use
    
    AUTOTUNE is used at dynamic run time for better performance
    IMAGE_SIZE could be scaled to improve efficiency 
               orignal image size 208 X 176
"""

EPOCHS = 25
IMAGE_SIZE = [208, 176]
NUM_CLASSES = 4
BATCH = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dir = '/Users/ellie/Downloads/Alzheimer_s Dataset/train/'
test_dir = '/Users/ellie/Downloads/Alzheimer_s Dataset/test/'


"""
    Accessing training data subdirectories from local computer by main directory
         
    subset: Splitting the training data to 'train'set and 'validation' set 
    with 80:20 ratio by random shuffling 

    color_mode: All images were set to grayscale with 1 channel ("rgb" for 3 channels )

    seed: allow for reproducability
"""

train = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, # the training data directory
    labels = "inferred", # gets data labels from directory structure
    label_mode = 'categorical',
    color_mode = "grayscale",
    batch_size = BATCH,
    image_size = IMAGE_SIZE,
    seed = 1,
    validation_split = 0.2,
    subset = "training",
    
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels = "inferred", 
    label_mode = 'categorical',
    color_mode = "grayscale",
    batch_size = BATCH,
    image_size = IMAGE_SIZE,
    seed = 1,
    validation_split = 0.2,
    subset = "validation",
)

"""
    Dataset.cache(): keep images in memory to improve model training efficiency
    Dataset.prefetch(): overlaps data preprocessing and model execution to improve 
                        model training efficiency
"""

train = train.cache().prefetch(buffer_size=AUTOTUNE) 
validation = validation.cache().prefetch(buffer_size=AUTOTUNE) 

"""
    The model structure: we use the seqential model with 6 hidden layers, 7 layers total
"""

model = tf.keras.Sequential([
    
    # input layer
    tf.keras.Input(shape=(*IMAGE_SIZE, 1)),

    # hidden layer 1 
    tf.keras.layers.SeparableConv2D(128, 7, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    
    # hidden layer 2
    tf.keras.layers.SeparableConv2D(64, 7, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    # hidden layer 3
    tf.keras.layers.SeparableConv2D(64, 7, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # hidden layer 4
    tf.keras.layers.SeparableConv2D(32, 7, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    # hidden layer 5
    tf.keras.layers.SeparableConv2D(16, 7, activation='relu', padding='same'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    #hidden layer 6
    tf.keras.layers.SeparableConv2D(16, 7, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Flatten(),
    #output layer with softmax
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


"""
    A weighted version of keras.objectives.categorical_crossentropy
    We found this function from an online resource: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852dthat
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([4, 50, 1, 1.4]) -> our own weights
        Class one at 4, class 2 50x the normal weights, class 4 1.4x (due to data imbalance).
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
"""
def weighted_categorical_crossentropy(weights):
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calculate loss
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

# setting the weights for weighted cross entropy
weights = np.array([4, 50, 1, 1.4])

# compiling and training the model
model.compile(
        optimizer = 'adam',
        loss = weighted_categorical_crossentropy(weights),
        metrics = [tf.keras.metrics.AUC(name='auc'), 
           tf.keras.metrics.CategoricalAccuracy(name="cat acc"),
           tf.keras.metrics.Recall(name="recall")],
    )

history = model.fit(
   train,
   validation_data = validation,
   epochs = EPOCHS
)

'''
graphs for AUC, loss, categorical accuracy over epochs
'''

# AUC
plt.figure()
plt.plot(history.history['auc'], label = "train")
plt.plot(history.history['val_auc'], label = "validation")
plt.title('Area under the curve')
plt.xlabel('epochs')
plt.ylabel('AUC')
plt.legend()
plt.savefig('auc')

# Loss
plt.figure()
plt.plot(history.history['loss'], label = "train")
plt.plot(history.history['val_loss'], label = "validation")
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss')

# Categorical Accuracy
plt.figure()
plt.plot(history.history['cat acc'], label = "train")
plt.plot(history.history['val_cat acc'], label = "validation")
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('acc')


'''  Testing the model '''

# Retrieving test data set
test = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, # the test data directory
    labels = "inferred", 
    label_mode = 'categorical',
    color_mode = "grayscale",
    image_size = IMAGE_SIZE,
)

test = test.cache().prefetch(buffer_size=AUTOTUNE)
    
model.evaluate(test) # running model on test set
model.summary()
#saving the model
model.save('Alzheimer')