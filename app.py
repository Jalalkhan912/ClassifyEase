from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from PIL import Image  # Import Image from PIL
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


IMG_WIDTH, IMG_HEIGHT = 128, 128


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_images', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    class_index = request.form.get('class_index')
    class_name = request.form.get('class_name')
    
    if not class_index or not class_name:
        return jsonify({'success': False, 'error': 'Class information missing'})

    class_folder = os.path.join(UPLOAD_FOLDER, f'class_{class_index}_{class_name}')
    os.makedirs(class_folder, exist_ok=True)

    files = request.files.getlist('images')
    count = 0
    for file in files:
        if file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(class_folder, filename))
            count += 1

    return jsonify({'success': True, 'count': count})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        num_classes = int(request.form['num_classes'])
        class_names = [request.form.get(f'class_name_{i}') for i in range(1, num_classes + 1) if request.form.get(f'class_name_{i}')]
        
        if len(class_names) != num_classes:
            return jsonify({'error': 'Mismatch in number of classes and provided class names'}), 400

        class_images = {name: [os.path.join(UPLOAD_FOLDER, f'class_{i+1}_{name}', f) 
                               for f in os.listdir(os.path.join(UPLOAD_FOLDER, f'class_{i+1}_{name}'))]
                        for i, name in enumerate(class_names)}

        if not all(class_images.values()):
            return jsonify({'error': 'No images found for one or more classes'}), 400

        num_conv_layers = int(request.form['conv_layers'])
        filters = [int(request.form[f'filters_{i}']) for i in range(1, num_conv_layers + 1)]

        model = build_model(num_classes, num_conv_layers, filters)
        X_train, X_test, y_train, y_test = preprocess_images(class_images, num_classes)
        history = train_model(X_train, y_train, X_test, y_test, model)
        
        model.save('uploads/trained_model.h5')  # Save model after training

        return jsonify(return_training_results(history))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['POST'])
def test_model():
    try:
        # Get test images from the request
        test_images = request.files.getlist('test_images')
        
        # Check if any images were uploaded
        if len(test_images) == 0:
            return jsonify({"error": "No test images uploaded."})

        # Preprocess the test images
        processed_images = []
        for img in test_images:
            image = Image.open(img)
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to match model input
            image = img_to_array(image) / 255.0  # Normalize if necessary
            processed_images.append(image)

        # Convert list to numpy array
        processed_images = np.array(processed_images)

        # Load the trained model
        model = tf.keras.models.load_model('uploads/trained_model.h5')

        # Dynamically determine class names from the upload directory
        class_names = []
        for folder_name in os.listdir(UPLOAD_FOLDER):
            if os.path.isdir(os.path.join(UPLOAD_FOLDER, folder_name)):
                class_names.append(folder_name)

        # Ensure class names are available
        if not class_names:
            return jsonify({"error": "No class names found. Ensure that the model was trained with proper class directories."})

        # Predict using the model
        predictions = model.predict(processed_images)

        # Process predictions
        results = []
        for idx, prediction in enumerate(predictions):
            predicted_class_index = np.argmax(prediction)  # Get class with highest probability
            results.append({
                "image": test_images[idx].filename,
                "class": class_names[predicted_class_index]  # Map to class name
            })

        # Return the results as JSON
        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)})

def load_model():
    # Load the saved model
    return tf.keras.models.load_model('uploads/trained_model.h5')

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

def preprocess_images(class_images, num_classes, img_size=(IMG_WIDTH, IMG_HEIGHT)):
    X, y = [], []
    for class_index, (class_name, image_paths) in enumerate(class_images.items()):
        for image_path in image_paths:
            image = load_img(image_path, target_size=img_size)
            image = img_to_array(image) / 255.0
            X.append(image)
            y.append(class_index)

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test, model, epochs=10, batch_size=32):
    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     epochs=epochs, batch_size=batch_size)

def return_training_results(history):
    return {
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_loss': float(history.history['loss'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
        'val_loss': float(history.history['val_loss'][-1])
    }

if __name__ == '__main__':
    app.run(debug=True)
