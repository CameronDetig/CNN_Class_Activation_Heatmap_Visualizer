import os
os.environ["KERAS_BACKEND"] = "jax"

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
import keras_hub
import numpy as np
import jax
from keras import ops
from PIL import Image

# Global variables for models
model = None
last_conv_layer_model = None
classifier_model = None

def initialize_models():
    """Initialize the models once when the app starts."""
    global model, last_conv_layer_model, classifier_model
    
    # Load the pretrained Xception model
    model = keras_hub.models.ImageClassifier.from_preset(
        "xception_41_imagenet",
        activation="softmax",
    )
    
    # Create a model that maps the input image to the activations of the last convolutional layer
    last_conv_layer_name = "block14_sepconv2_act"
    last_conv_layer = model.backbone.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    
    # Create a model that maps the activations of the last convolutional layer to the final class predictions
    classifier_input = last_conv_layer.output
    x = classifier_input
    for layer_name in ["pooler", "predictions"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

def loss_fn(last_conv_layer_output):
    """Defines a separate loss function for gradient computation."""
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = ops.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]
    return top_class_channel[0]

# Create gradient function
grad_fn = jax.grad(loss_fn)

def get_top_class_gradients(img_array):
    """Get gradients of the top predicted class with respect to last conv layer."""
    last_conv_layer_output = last_conv_layer_model(img_array)
    grads = grad_fn(last_conv_layer_output)
    return grads, last_conv_layer_output

def generate_heatmap(image):
    """
    Generate class activation heatmap for an uploaded image.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        tuple: (superimposed_img, prediction_text)
    """
    if image is None:
        return None, "Please upload an image."
    
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Prepare image for model (add batch dimension)
    img_array = np.expand_dims(img, axis=0)
    
    # Get predictions
    preds = model.predict(img_array, verbose=0)
    
    # Decode predictions
    decoded_preds = keras_hub.utils.decode_imagenet_predictions(preds)
    
    # Format prediction text
    prediction_text = "Top 5 Predictions:\n\n"
    for i, (description, score) in enumerate(decoded_preds[0][:5], 1):
        prediction_text += f"{i}. {description}: {score:.2%}\n"
    
    # Preprocess image
    img_array = model.preprocessor(img_array)
    
    # Get gradients and last conv layer output
    grads, last_conv_layer_output = get_top_class_gradients(img_array)
    grads = ops.convert_to_numpy(grads)
    last_conv_layer_output = ops.convert_to_numpy(last_conv_layer_output)
    
    # Compute importance of each channel
    pooled_grads = np.mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0].copy()
    
    # Weight each channel by its importance
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Apply jet colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Convert to image and resize to match original
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    
    # Superimpose heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    
    return superimposed_img, prediction_text

# Initialize models when the script loads
print("Initializing models... this may take a moment.")
initialize_models()
print("Models initialized!")

# Create Gradio interface
with gr.Blocks(title="Class Activation Heatmap Visualizer") as demo:
    gr.Markdown(
        """
        # Class Activation Heatmap Visualizer
        
        Upload an image or choose one of the examples to see what parts of the image the neural network focuses on when making predictions.
        The heatmap shows which regions of the image are most important for the top predicted class.

        Code adapted from: https://deeplearningwithpython.io/chapters/chapter10_interpreting-what-convnets-learn/#visualizing-heatmaps-of-class-activation
        
        **Model:** Xception trained on ImageNet (1,000 classes)
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
            submit_btn = gr.Button("Generate Heatmap", variant="primary", size="lg")

            # Example images
            gr.Examples(
                examples=[
                    ["images/elephant.jpg"],
                    ["images/dog.jpg"],
                    ["images/F1_car.jpg"],
                    ["images/multiple_animals.jpg"],
                    ["images/osprey.jpeg"]
                ],
                inputs=input_image,
                label="Try an example:"
            )

            gr.Markdown(
                """
                ### How to interpret the heatmap:
                - **Red/Yellow regions**: Areas the model focuses on most for its prediction
                - **Blue/Purple regions**: Areas the model considers less important
                """
            )
            
        with gr.Column():
            output_image = gr.Image(
                label="Heatmap Visualization",
                type="pil",
                height=400
            )
            prediction_text = gr.Textbox(
                label="Predictions",
                lines=7,
                interactive=False
            )
    
    # Connect the button to the function
    submit_btn.click(
        fn=generate_heatmap,
        inputs=input_image,
        outputs=[output_image, prediction_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
