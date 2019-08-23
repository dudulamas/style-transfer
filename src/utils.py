import tensorflow as tf
from tensorflow.python.keras import models 
from tensorflow.python.keras.preprocessing import image as kp_image

import matplotlib.pyplot as plt
import numpy as np

def enable_tf_eager(tf):
    tf.enable_eager_execution()
    print("Eager execution: {}".format(tf.executing_eagerly()))

def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def load_img(path):
    from PIL import Image
    
    max_dim = 512
    img = Image.open(path)
    long = max(img.size)
    scale = max_dim/long
    
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)    
    img = kp_image.img_to_array(img)    
    img = np.expand_dims(img, axis=0)
    return img

def show_img(img, title=None):
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    
    # Normalize for display 
    out = out.astype('uint8')
    plt.imshow(out)
    
    if title is not None:
        plt.title(title)

    plt.imshow(out)

def get_model(style_layers, content_layers):
    """ 
    Esta função carregará o modelo VGG19 e acessará as suas camadas intermediárias. 

    Carregaremos nossa rede de classificação de imagem pré-treinada. 
    Então pegamos as camadas de interesse como definimos anteriormente. 
    
    Em seguida, definimos um modelo definindo as entradas do modelo para uma imagem e as saídas 
    para as saídas das camadas de estilo e conteúdo. 
    
    Em outras palavras, criamos um modelo que pegará uma imagem de entrada e gerará o conteúdo 
    e estilizará camadas intermediárias!
    """
    # Load our model. We load pretrained VGG, trained on imagenet data (weights=’imagenet’)
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    print(">> Model outputs: ", model_outputs)

    return models.Model(vgg.input, model_outputs)

def  gram_matrix(input_tensor):
    channels =  int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor,[-1 , channels])
    n = tf.shape(a)[0]
    gram_a = tf.matmul(a, a, transpose_a=True)
    return gram_a / tf.cast(n, tf.float32)

def deprocess_img(processed_img):
    x = processed_img.copy()
    
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                "dimension [1, height, width, channel] or [height, width, channel]")
    
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x
