import streamlit as st
import pandas as pd

import os

import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="FishNET",
    page_icon=":fish:",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://google.com/',
        'Report a bug': "https://google.com/",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title("FishNET :fish: ")

st.subheader("Fisch im Netz und keine Ahnung, ob es sich um eine gefährdete Fischart handelt?")
st.info("**FishNET** ist eine Fischerkennungs-App, die die häufigsten in der Schweiz vorkommenden Fische mittels automatischer Bilderkennung bestimmt und vor gefährdeten Fischarten warnt.")
st.info("Foto aufnehmen oder hochladen und in Sekundenschnelle der gefangene Fisch mit **FishNET** schnell und praktisch bestimmen.")


#################################################################

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
#model.summary()
st.write(model.summary)

checkpoint_path = "trained_weights/trained_weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

os.listdir(checkpoint_dir)

# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
st.write("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
st.write("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "trained_weights_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

os.listdir(checkpoint_dir)

latest = tf.train.latest_checkpoint(checkpoint_dir)
st.write(latest)

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
st.write("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Save the weights
model.save_weights('checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
st.write("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
#!mkdir -p saved_model
model.save('saved_model/my_model')

new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
#new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

#print(new_model.predict(test_images).shape)

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')


# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
#new_model.summary()
st.write("new_model.summary: ")
st.write(new_model.summary)

#################################################################




#st.image('olaf.jpg')

# Save the weights
model = create_model()

# Create a new model instance
model.load_weights('trained_weights/trained_weights.ckpt')

# kei ahnig, was mached mir do?
#print()
#st.write(model.predict('olaf.jpg').shape)

with st.expander("Foto hochladen"):
    file = st.file_uploader("")

with st.expander("Foto aufnehmen"):
    picture = st.camera_input("")

st.write("")