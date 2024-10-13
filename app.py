import streamlit as st
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import io

# Constants
IMG_WIDTH, IMG_HEIGHT = 128, 128
UPLOAD_FOLDER = 'uploads/'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Streamlit app
def main():
    st.title("ClassifyEase - Model Training and Testing")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Train Model", "Test Model"])
    
    if page == "Train Model":
        train_model_page()
    else:
        test_model_page()

def train_model_page():
    st.header("Train Your Model")
    
    # Number of classes
    num_classes = st.number_input("Number of Classes", min_value=2, value=2, step=1)
    
    # Class names and image upload
    class_data = {}
    for i in range(1, num_classes + 1):
        st.subheader(f"Class {i}")
        class_name = st.text_input(f"Class {i} Name", key=f"class_name_{i}")
        uploaded_files = st.file_uploader(f"Upload images for Class {i}", accept_multiple_files=True, key=f"class_images_{i}")
        
        if class_name and uploaded_files:
            class_folder = os.path.join(UPLOAD_FOLDER, f'class_{i}_{class_name}')
            os.makedirs(class_folder, exist_ok=True)
            
            for file in uploaded_files:
                img = Image.open(file)
                img.save(os.path.join(class_folder, file.name))
            
            class_data[class_name] = [os.path.join(class_folder, f.name) for f in uploaded_files]
    
    # Model architecture
    st.subheader("Model Architecture")
    num_conv_layers = st.number_input("Number of Conv Layers", min_value=1, max_value=5, value=2, step=1)
    filters = [st.number_input(f"Filters for Conv Layer {i+1}", min_value=16, max_value=256, value=32, step=16) for i in range(num_conv_layers)]
    
    # Training parameters
    st.subheader("Training Parameters")
    epochs = st.number_input("Number of Epochs", min_value=1, value=10, step=1)
    batch_size = st.number_input("Batch Size", min_value=1, value=32, step=1)
    
    # Training button
    if st.button("Start Training"):
        if len(class_data) != num_classes:
            st.error("Please upload images for all classes before training.")
        else:
            with st.spinner("Training in progress..."):
                model = build_model(num_classes, num_conv_layers, filters)
                X_train, X_test, y_train, y_test = preprocess_images(class_data, num_classes)
                
                # Create progress bar and metrics
                progress_bar = st.progress(0)
                epoch_text = st.empty()
                metrics_text = st.empty()
                
                # Custom callback to update Streamlit components
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress_bar.progress((epoch + 1) / epochs)
                        epoch_text.text(f"Epoch {epoch + 1}/{epochs}")
                        metrics_text.text(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
                                          f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")
                
                history = train_model(X_train, y_train, X_test, y_test, model, epochs, batch_size, StreamlitCallback())
                model.save(os.path.join(UPLOAD_FOLDER, 'trained_model.h5'))
                
                st.success("Training complete!")
                
                # Display final metrics
                st.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
                st.write(f"Final training loss: {history.history['loss'][-1]:.4f}")
                st.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
                st.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
                
                # Plot and display training history
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                ax1.plot(history.history['accuracy'], label='Training Accuracy')
                ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                
                ax2.plot(history.history['loss'], label='Training Loss')
                ax2.plot(history.history['val_loss'], label='Validation Loss')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                
                st.pyplot(fig)
                
                # Download button for the trained model
                model_path = os.path.join(UPLOAD_FOLDER, 'trained_model.h5')
                with open(model_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Trained Model",
                        data=file,
                        file_name="trained_model.h5",
                        mime="application/octet-stream"
                    )

def test_model_page():
    st.header("Test Your Model")
    
    uploaded_files = st.file_uploader("Upload test images", accept_multiple_files=True)
    
    if uploaded_files and st.button("Run Test"):
        model_path = os.path.join(UPLOAD_FOLDER, 'trained_model.h5')
        if not os.path.exists(model_path):
            st.error("No trained model found. Please train a model first.")
        else:
            model = tf.keras.models.load_model(model_path)
            
            results = []
            for file in uploaded_files:
                img = Image.open(file)
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array)
                class_idx = np.argmax(prediction[0])
                
                # Get class names from the upload directory
                class_names = [name for name in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, name)) and name.startswith('class_')]
                class_names.sort(key=lambda x: int(x.split('_')[1]))
                
                results.append({
                    "image": file.name,
                    "class": class_names[class_idx].split('_', 2)[-1]
                })
            
            st.subheader("Test Results")
            for result in results:
                st.write(f"{result['image']}: Classified as {result['class']}")

def build_model(num_classes, num_conv_layers, filters):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    for filter_size in filters:
        model.add(tf.keras.layers.Conv2D(filter_size, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_images(class_images, num_classes):
    X, y = [], []
    for class_index, (class_name, image_paths) in enumerate(class_images.items()):
        for image_path in image_paths:
            image = Image.open(image_path)
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image = img_to_array(image) / 255.0
            X.append(image)
            y.append(class_index)

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test, model, epochs, batch_size, callback):
    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     epochs=epochs, batch_size=batch_size, callbacks=[callback])

if __name__ == '__main__':
    main()