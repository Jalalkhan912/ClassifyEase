# ClassifyEase - Custom Image Classification Made Easy

A user-friendly web application that allows you to train custom image classification models without any coding knowledge. Built with Streamlit and TensorFlow, ClassifyEase provides an intuitive interface for creating, training, and testing your own CNN models.

## 🌟 Features

### 🚀 **No-Code Model Training**
- Upload images for multiple classes
- Customize model architecture with configurable CNN layers
- Real-time training progress visualization
- Automatic model saving and downloading

### 🎯 **Flexible Configuration**
- Support for 2+ custom classes
- Adjustable number of convolutional layers (1-5)
- Configurable filter sizes (16-256)
- Customizable training parameters (epochs, batch size)

### 📊 **Real-Time Monitoring**
- Live training progress bars
- Real-time loss and accuracy metrics
- Interactive training history plots
- Validation performance tracking

### 🧪 **Easy Model Testing**
- Upload test images for instant predictions
- Batch image classification
- Clear result visualization

## 🏗️ Architecture

ClassifyEase uses a flexible CNN architecture that adapts to your specifications:

- **Input Layer**: 128x128x3 RGB images
- **Convolutional Layers**: Configurable (1-5 layers)
- **Pooling**: MaxPooling2D after each conv layer
- **Dense Layers**: 128 neurons + ReLU activation
- **Output Layer**: Softmax for multi-class classification

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jalalkhan912/ClassifyEase.git
   cd ClassifyEase
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📋 Requirements

```
tensorflow
Pillow
numpy
scikit-learn
matplotlib
```

## 📖 How to Use

### 🎓 Training a Model

1. **Navigate to "Train Model" page** from the sidebar
2. **Set number of classes** you want to classify
3. **For each class:**
   - Enter a descriptive class name
   - Upload multiple images (JPG, PNG supported)
4. **Configure model architecture:**
   - Choose number of convolutional layers (1-5)
   - Set filter sizes for each layer (16-256)
5. **Set training parameters:**
   - Number of epochs (recommended: 10-50)
   - Batch size (recommended: 16-64)
6. **Click "Start Training"** and monitor progress
7. **Download your trained model** when complete

### 🔍 Testing a Model

1. **Navigate to "Test Model" page** from the sidebar
2. **Upload test images** you want to classify
3. **Click "Run Test"** to get predictions
4. **View results** showing predicted classes for each image

## 💡 Best Practices

### 📸 **Image Preparation**
- Use high-quality images (minimum 128x128 pixels)
- Ensure good lighting and clear subjects
- Include diverse examples for each class
- Aim for at least 20-50 images per class

### 🏛️ **Model Architecture**
- Start with 2 conv layers for simple tasks
- Use more layers (3-5) for complex image features
- Increase filter sizes (64, 128, 256) for detailed recognition
- Monitor validation accuracy to avoid overfitting

### ⚙️ **Training Parameters**
- Start with 10-20 epochs for initial testing
- Use batch size 32 for balanced memory usage
- Monitor validation loss - stop if it starts increasing

## 📁 Project Structure

```
classifyease/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── uploads/              # Auto-created directory for:
│   ├── class_1_name/     # Class 1 training images
│   ├── class_2_name/     # Class 2 training images
│   ├── ...               # Additional class folders
│   └── trained_model.h5  # Saved model file
└── README.md            # Project documentation
```

## 🎨 Use Cases

### 🏥 **Medical Imaging**
- X-ray classification (normal vs. abnormal)
- Skin lesion detection
- Microscopy image analysis

### 🌾 **Agriculture**
- Plant disease identification
- Crop type classification
- Pest detection

### 🏭 **Manufacturing**
- Quality control inspection
- Defect detection
- Product categorization

### 🎨 **Creative Applications**
- Art style classification
- Object recognition for inventory
- Custom visual search systems

## ⚡ Performance Tips

### 🖼️ **Image Optimization**
- Resize large images before upload to speed up training
- Use consistent image formats (JPG/PNG)
- Ensure images are properly oriented

### 🧠 **Model Optimization**
- Start simple and gradually increase complexity
- Use data augmentation for small datasets
- Consider transfer learning for complex tasks

## 🔧 Advanced Configuration

### Custom Image Size
To change the default image size, modify these constants in `app.py`:
```python
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Change from default 128, 128
```

### Model Architecture Customization
The model architecture can be extended by modifying the `build_model()` function:
```python
# Add dropout layers
model.add(tf.keras.layers.Dropout(0.5))

# Add batch normalization
model.add(tf.keras.layers.BatchNormalization())
```

## 📊 Technical Specifications

- **Input Format**: RGB images, automatically resized to 128x128
- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Frontend**: Streamlit
- **Data Split**: 80% training, 20% validation
- **Optimization**: Adam optimizer with categorical crossentropy loss

## 🐛 Troubleshooting

### Common Issues

**"No trained model found" error:**
- Ensure you've completed training before testing
- Check that `trained_model.h5` exists in the uploads folder

**Out of memory errors:**
- Reduce batch size (try 16 or 8)
- Use fewer/smaller images for training
- Reduce number of conv layers or filters

**Poor model performance:**
- Increase number of training images per class
- Add more epochs to training
- Ensure image quality and diversity
- Check class balance (similar number of images per class)

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 🎯 Areas for Contribution
- Data augmentation features
- Transfer learning integration
- Model export to different formats
- Advanced visualization tools
- Mobile-friendly interface

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** - For the powerful ML framework
- **Streamlit Team** - For the intuitive web app framework
- **scikit-learn** - For preprocessing utilities
- **PIL/Pillow** - For image processing capabilities

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Jalalkhan912/ClassifyEase/issues)
- **Email**: jalalkhanscience@gmail.com
  
## 🚀 Future Roadmap

- [ ] Transfer learning with pre-trained models
- [ ] Data augmentation options
- [ ] Model comparison tools
- [ ] Export to TensorFlow Lite
- [ ] Batch prediction API
- [ ] Integration with cloud storage
- [ ] Advanced hyperparameter tuning

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/)
