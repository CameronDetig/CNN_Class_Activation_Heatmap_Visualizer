# CNN Class Activation Heatmap Visualizer
Hugging Face Space to display a heatmap of areas a CNN uses for classification. Utilizes Keras with the JAX backend and Gradio for the UI.

Try it out! You may need to restart the space if it is asleep: https://huggingface.co/spaces/cameron-d/CNN_Class_Activation_Heatmap_Visualizer

Code adapted from: https://deeplearningwithpython.io/chapters/chapter10_interpreting-what-convnets-learn/#visualizing-heatmaps-of-class-activation


Model Used: Xception trained on ImageNet (1,000 classes) https://www.kaggle.com/models/keras/xception/keras/xception_41_imagenet/2

![Heatmap Example](images/hugging_face_UI.png)



## To run locally:
Clone the repo and run ```pip install -r requirements.txt```

Run app.py

A link in the terminal will appear. Click that to open the Gradio UI in a browser.