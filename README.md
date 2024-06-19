## Human_Body_Detection_Model: Detecting Human-Like Figures with a Fine-Tuned ResNet-50

This repository contains the code and resources for a human body detection model trained on a dataset of self-classified and captured images of a human-like doll.

**Model Architecture:**
- The model utilizes a pre-trained ResNet-50 architecture as the foundation. 
- A Global Average Pooling layer is added to capture the global features of the input image.
- A Dense layer with a ReLU (Rectified Linear Unit) activation function is then added for further feature extraction.
- Dropout with a rate of 0.5 is used to prevent overfitting during training.
- Finally, a single output neuron with a sigmoid activation function predicts the presence of a human-like figure in the image (sigmoid output closer to 1 indicates higher probability of a human being present).

**Model File:**
- The trained model is saved as `model.h5` and can be loaded using libraries like TensorFlow or Keras.

**Dataset:**
- The dataset used for training is included in this repository.

**Getting Started:**
1. Ensure you have the necessary libraries installed (e.g., TensorFlow, Keras).
2. Load the pre-trained model using `model = tf.keras.models.load_model('model.h5')`.
3. Pre-process your input image according to the format expected by the model (likely resizing and normalization).
4. Use `model.predict(preprocessed_image)` to get the probability of a human-like figure being present in the image.

**Disclaimer:**
- This model is trained on a limited dataset of doll images and may not perform optimally on all real-world scenarios with diverse human body shapes and poses.
- Further training on a more comprehensive dataset is recommended for improved accuracy and generalizability.

**Feel free to explore the code and adapt it to your specific needs!**
