import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import random

def show_predicted_img(model,test_images_directory,class_name_dict):

    
    # Prepare image paths and their true labels
    image_paths = []
    true_labels = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(test_images_directory):
        for file in files:        
            if file.lower().endswith('.png') or file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):            
                image_paths.append(os.path.join(root, file))            
                true_labels.append(root.split(os.path.sep)[-1])  # Assuming folder name is the true label
    
    # Randomly select a subset of images
    num_images_to_show = 5
    selected_indices = random.sample(range(len(image_paths)), num_images_to_show)
    selected_image_paths = [image_paths[i] for i in selected_indices]
    selected_true_labels = [true_labels[i] for i in selected_indices]

    # Prepare images for prediction
    images_for_prediction = []
    for image_path in selected_image_paths:    
        image = load_img(image_path, target_size=(224, 224))  # Change target_size to match your model's expected input    
        image_array = img_to_array(image)    
        images_for_prediction.append(image_array)
    
    # Convert to a numpy array and scale the images if necessary (model dependent)
    images_for_prediction = np.array(images_for_prediction)

    # Make predictions
    predictions = model.predict(images_for_prediction)

    # Display images with predictions and true labels
    fig, axs = plt.subplots(1, num_images_to_show, figsize=(20, 5))  
    # Adjust the figure size as needed
    for i, (image, prediction, true_label) in enumerate(zip(images_for_prediction, predictions, selected_true_labels)):    
        axs[i].imshow(image.astype(np.uint8))    
        axs[i].axis('off')  # Hide the axes    
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability class
        
        probability = np.max(prediction)  # Get the probability of the predicted class    
        axs[i].set_title(f'Predicted: {class_name_dict[predicted_class]}\nProbability: {probability:.2f}\nTrue: {true_label}')
        # Show the plot
    plt.tight_layout()
    plt.show()