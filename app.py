from flask import Flask, request, render_template, send_file
import os
import cv2
import io
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from imageio import mimsave
from keras import backend as K
import matplotlib.pyplot as plt
from IPython.display import display as display_fn
from IPython.display import Image, clear_output
app = Flask(__name__)

# Your style transfer functions go here
# (Use the provided style transfer functions and classes from the previous code snippet)

def tensor_to_image(tensor):
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)

def load_img(path_to_img):
    max_dim = 512
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image

def preprocess_image(image):
    image = tf.cast(image, dtype=tf.float32)
    image = (image / 127.5) - 1.0
    return image

def clip_image_values(image, min_value=0.0, max_value=255.0):
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)

def get_style_image_features(image, model):  
    preprocessed_style_image = preprocess_image(image) 
    outputs = model(preprocessed_style_image) 
    style_outputs = outputs[:NUM_STYLE_LAYERS] 
    gram_style_features = [gram_matrix(style_layer) for style_layer in style_outputs] 
    return gram_style_features

def get_content_image_features(image, model):
    preprocessed_content_image = preprocess_image(image)
    outputs = model(preprocessed_content_image) 
    content_outputs = outputs[NUM_STYLE_LAYERS:]
    return content_outputs

def gram_matrix(input_tensor):
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) 
    input_shape = tf.shape(input_tensor) 
    height = input_shape[1] 
    width = input_shape[2] 
    num_locations = tf.cast(height * width, tf.float32)
    scaled_gram = gram / num_locations
    return scaled_gram

def get_style_content_loss(style_targets, style_outputs, content_targets, content_outputs, style_weight, content_weight):
    style_loss = tf.add_n([get_style_loss(style_output, style_target) for style_output, style_target in zip(style_outputs, style_targets)])
    content_loss = tf.add_n([get_content_loss(content_output, content_target) for content_output, content_target in zip(content_outputs, content_targets)])
    style_loss = style_loss * style_weight / NUM_STYLE_LAYERS 
    content_loss = content_loss * content_weight / NUM_CONTENT_LAYERS 
    total_loss = style_loss + content_loss 
    return total_loss

def calculate_gradients(image, style_targets, content_targets, style_weight, content_weight, var_weight, model):
    with tf.GradientTape() as tape:
        style_features = get_style_image_features(image, model) 
        content_features = get_content_image_features(image, model) 
        loss = get_style_content_loss(style_targets, style_features, content_targets, content_features, style_weight, content_weight) 
    gradients = tape.gradient(loss, image) 
    return gradients

def update_image_with_style(image, style_targets, content_targets, style_weight, var_weight, content_weight, optimizer, model):
    gradients = calculate_gradients(image, style_targets, content_targets, style_weight, content_weight, var_weight, model) 
    optimizer.apply_gradients([(gradients, image)]) 
    image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))

def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4, var_weight=0, optimizer='adam', epochs=1, steps_per_epoch=1):
    '''Fits the style transfer model'''
    images = []
    step = 0
    style_targets = get_style_image_features(style_image)
    content_targets = get_content_image_features(content_image)
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image) 
    images.append(content_image)
    image = tensor_to_image(content_image)
    display_fn(image)
    
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            update_image_with_style(generated_image, style_targets, content_targets, style_weight, var_weight, content_weight, optimizer)
            if (m + 1) % 10 == 0:
                images.append(generated_image)
        clear_output(wait=True)
        display_image = tensor_to_image(generated_image)
        display_fn(display_image)
        images.append(generated_image)
        print(f"Train step: {step}")
        
    generated_image = tf.cast(generated_image, dtype=tf.uint8)
    return generated_image, images

# Modify the parameters for faster execution


# Define style and content weight
style_weight = 1e-1
content_weight = 1e-32
INITIAL_LEARNING_RATE = 80.0
DECAY_STEPS = 100
DECAY_RATE = 0.80

# Define optimizer
adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE
    )
)

style_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4'] 
content_layers = ['conv2d_88'] 
output_layers = style_layers + content_layers 
NUM_CONTENT_LAYERS = len(content_layers)
NUM_STYLE_LAYERS = len(style_layers)
def inception_model(layer_names):
    # Load the pretrained InceptionV3, trained on imagenet data
    inception = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
    # Freeze the weights of the model's layers (make them not trainable)
    inception.trainable = False
    # Create a list of layer objects that are specified by layer_names
    outputs = [inception.get_layer(name).output for name in layer_names]
    # Create the model that outputs content and style layers only
    model = tf.keras.Model(inputs=inception.input, outputs=outputs)
    return model
K.clear_session()
inception = inception_model(output_layers)

@app.route('/')
def index():
    return render_template('index.html')
def get_style_loss(features, targets):
    '''Gets the style loss between features and targets'''
    return tf.reduce_mean(tf.square(features - targets))
def get_content_loss(features, targets):
    '''Gets the content loss between features and targets'''
    return 0.5 * tf.reduce_sum(tf.square(features - targets))
def gram_matrix(input_tensor):
    '''Calculates the gram matrix of the input tensor'''
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) 
    input_shape = tf.shape(input_tensor) 
    height = input_shape[1] 
    width = input_shape[2] 
    num_locations = tf.cast(height * width, tf.float32)
    scaled_gram = gram / num_locations
    return scaled_gram
def get_style_image_features(image):  
    '''Gets the style features of the image'''
    preprocessed_style_image = preprocess_image(image) 
    outputs = inception(preprocessed_style_image) 
    style_outputs = outputs[:NUM_STYLE_LAYERS] 
    gram_style_features = [gram_matrix(style_layer) for style_layer in style_outputs] 
    return gram_style_features
def get_content_image_features(image):
    '''Gets the content features of the image'''
    preprocessed_content_image = preprocess_image(image)
    outputs = inception(preprocessed_content_image) 
    content_outputs = outputs[NUM_STYLE_LAYERS:]
    return content_outputs
def get_style_content_loss(style_targets, style_outputs, content_targets, content_outputs, style_weight, content_weight):
    '''Calculates the total loss'''
    style_loss = tf.add_n([get_style_loss(style_output, style_target) for style_output, style_target in zip(style_outputs, style_targets)])
    content_loss = tf.add_n([get_content_loss(content_output, content_target) for content_output, content_target in zip(content_outputs, content_targets)])
    style_loss = style_loss * style_weight / NUM_STYLE_LAYERS 
    content_loss = content_loss * content_weight / NUM_CONTENT_LAYERS 
    total_loss = style_loss + content_loss 
    print(f"Total Loss = {total_loss.numpy()} | Content Loss = {content_loss.numpy()} | Style Loss = {style_loss.numpy()}")
    return total_loss
def calculate_gradients(image, style_targets, content_targets, style_weight, content_weight, var_weight):
    '''Calculates gradients of the loss with respect to the input image'''
    with tf.GradientTape() as tape:
        style_features = get_style_image_features(image) 
        content_features = get_content_image_features(image) 
        loss = get_style_content_loss(style_targets, style_features, content_targets, content_features, style_weight, content_weight) 
    gradients = tape.gradient(loss, image) 
    return gradients
def update_image_with_style(image, style_targets, content_targets, style_weight, var_weight, content_weight, optimizer):
    '''Updates the image with the style'''
    gradients = calculate_gradients(image, style_targets, content_targets, style_weight, content_weight, var_weight) 
    optimizer.apply_gradients([(gradients, image)]) 
    image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))

# Modify the parameters for faster execution



@app.route('/upload', methods=['POST'])
def upload():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return 'No file uploaded', 400

    content_file = request.files['content_image']
    style_file = request.files['style_image']

    content_path = os.path.join('uploads', 'content.jpg')
    style_path = os.path.join('uploads', 'style.jpg')

    content_file.save(content_path)
    style_file.save(style_path)

    content_image = load_img(content_path)
    style_image = load_img(style_path)
    epochs = 1
    steps_per_epoch = 30
    style_image = tf.image.resize(style_image, [256, 256])
    content_image = tf.image.resize(content_image, [256, 256])
    stylized_image, _ = fit_style_transfer(style_image=style_image, content_image=content_image, 
                                       style_weight=style_weight, content_weight=content_weight,
                                       var_weight=0, optimizer=adam, epochs=epochs, steps_per_epoch=steps_per_epoch)



    output_image = tensor_to_image(stylized_image)

    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format='JPEG')
    output_buffer.seek(0)

    return send_file(output_buffer, mimetype='image/jpeg', as_attachment=True, download_name='stylized_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)

