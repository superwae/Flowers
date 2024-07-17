import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Load the model
model_location = 'flower_classifier_model.keras'
if os.path.exists(model_location):
    loaded_model = tf.keras.models.load_model(model_location, custom_objects={'KerasLayer': hub.KerasLayer})
else:
    print(f"Error: Model file not found at {model_location}")
    exit(1)

def preprocess_image(image):
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))
    img /= 255.0
    return img.numpy()

def make_prediction(img_path, model, num_top_predictions=5):
    image = Image.open(img_path)
    img_array = np.asarray(image)
    processed_img = preprocess_image(img_array)
    processed_img = np.expand_dims(processed_img, axis=0)
    preds = model.predict(processed_img)
    top_probs, top_indices = tf.math.top_k(preds, k=num_top_predictions)
    top_probs = top_probs.numpy().flatten()
    top_indices = top_indices.numpy().flatten()
    top_classes = [str(index) for index in top_indices]
    return top_probs, top_classes

def load_class_names(label_file):
    with open(label_file, 'r') as file:
        class_labels = json.load(file)
    return class_labels

def visualize_predictions(img_path, model, class_names, num_top_predictions=5):
    probabilities, classes = make_prediction(img_path, model, num_top_predictions)
    image = Image.open(img_path)
    
    fig, (ax_img, ax_bar) = plt.subplots(figsize=(10, 5), ncols=2)
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title('Input Image')
    
    y_positions = np.arange(len(classes))
    ax_bar.barh(y_positions, probabilities, align='center')
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels([class_names[str(cls)] for cls in classes])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Probability')
    ax_bar.set_title('Top Predictions')
    
    plt.tight_layout()
    plt.show()

def execute(image_path, top_k, category_names_file):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        exit(1)
    
    if category_names_file:
        class_names = load_class_names(category_names_file)
    else:
        class_names = None
    
    probabilities, classes = make_prediction(image_path, loaded_model, top_k)
    
    print("Predictions for the image:")
    for i in range(top_k):
        class_name = class_names[str(classes[i])] if class_names else classes[i]
        print(f"{i+1}: {class_name} with probability {probabilities[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the class of a flower from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to a JSON file mapping labels to flower names')
    args = parser.parse_args()

    execute(args.image_path, args.top_k, args.category_names)
