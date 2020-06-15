import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os
import time

print('TF Eager Execution status: ' + str(tf.executing_eagerly()))

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ASSOCIATION = {
    'Van Gogh': ROOT_DIR + "/art_pieces/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    'Hokusai': ROOT_DIR + "/art_pieces/The_Great_Wave_off_Kanagawa.jpg",
    "Kandinsky": ROOT_DIR + "/art_pieces/Vassily_Kandinsky,_1913_-_Composition_7.jpg"
}


def load_img(path):
    max_dim = 512
    img = Image.open(path)
    scale = max_dim/max(img.size)
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img, axis=0)


def imshow(img, title=None):
    out = np.squeeze(img, axis=0).astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)  # TODO: Check if this should be plt.imshow or plt.show


def load_and_process_img(path):
    img = load_img(path)
    return tf.keras.applications.vgg19.preprocess_input(img)


def de_process_img(processed_img):
    copy = processed_img.copy()
    if len(copy.shape) == 4:
        copy = np.squeeze(copy, 0)
    assert len(copy.shape) == 3, ("Input to de-process image must be an image of dimension [1, height, width, channel] "
                                  "or [height, width, channel]")
    if len(copy.shape) != 3:
        raise ValueError('Invalid De-processing Image input')

    de_processed_img = invert_vgg_processing(copy)
    return np.clip(de_processed_img, 0, 255).astype('uint8')


# Inverse of pre-processing step (BGR=channels) and mean = [103.939, 116.779, 123.68]
def invert_vgg_processing(vgg_processed_img):
    vgg_processed_img[:, :, 0] += 103.939
    vgg_processed_img[:, :, 1] += 116.779
    vgg_processed_img[:, :, 2] += 123.68
    return vgg_processed_img[:, :, ::-1]


def get_model(style_layers, content_layers):
    """ Creates our model with access to intermediate layers.

      This function will load the VGG19 model and access the intermediate layers.
      These layers will then be used to create a new model that will take input image
      and return the outputs from these intermediate layers from the VGG model.

      Returns:
        returns a keras model that takes image inputs and outputs the style and
          content intermediate layers.
    """
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    # Test with changed include_top later TODO
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    # Style and Content layers combine to create model output layer
    return tf.keras.models.Model(vgg.input, style_outputs + content_outputs)


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # Create image channels
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    # height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_feature_representations(model, content_path, style_path, num_style_layers):
    """Helper function to compute our content and style feature representations.

    This function will simply load and preprocess both the content and style
    images from their path. Then it will feed them through the network to obtain
    the outputs of the intermediate layers.

    Arguments:
        model: The model that we are using.
        content_path: The path to the content image.
        style_path: The path to the style image
        num_style_layers: The relevant layers for style feature extraction

    Returns:
        returns the style features and the content features.
    """
    # Load in imgs
    content_img = load_and_process_img(content_path)
    style_img = load_and_process_img(style_path)

    # Batch compute content and style features && get style and content feature reps from the model
    content_features = [content_layer[0] for content_layer in model(content_img)[num_style_layers:]]
    style_features = [style_layer[0] for style_layer in model(style_img)[:num_style_layers]]
    return style_features, content_features


# TODO: Computer loss with L-BFGS later
def compute_loss(model, loss_weights, init_img, gram_style_features,
                 content_features, num_style_layers, num_content_layers):
    """This function will compute the loss total loss.

      Arguments:
        model: The model that will give us access to the intermediate layers
        loss_weights: The weights of each contribution of each loss function.
          (style weight, content weight, and total variation weight)
        init_img: Our initial base image. This image is what we are updating with
          our optimization process. We apply the gradients wrt the loss we are
          calculating to this image.
        gram_style_features: Precomputed gram matrices corresponding to the
          defined style layers of interest.
        content_features: Precomputed outputs from defined content layers of
          interest.
        num_style_layers: The relevant number of layers for style feature extraction
        num_content_layers: The relevant number of layers for content feature extraction

      Returns:
        returns the total loss, style loss, content loss, and total variational loss
      """
    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_img)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score, content_score = 0, 0

    # Accumulate style losses from all layers
    # Equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= loss_weights[0]
    content_score *= loss_weights[1]

    # Returning Total loss
    return style_score + content_score, style_score, content_score


def compute_gradients(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    return tape.gradient(all_loss[0], cfg['init_img']), all_loss


def run_style_transfer(content_path, style_path, style_layers, content_layers,
                       num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)

    model = get_model(style_layers, content_layers)
    # Don't want or need to train any layers of our model
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path, num_style_layers)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # For displaying intermediate images
    # iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_img': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features,
        'num_style_layers': num_style_layers,
        'num_content_layers': num_content_layers
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_gradients(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = de_process_img(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()

            # Use the .numpy() method to get the concrete numpy array
            plot_img = init_image.numpy()
            plot_img = de_process_img(plot_img)
            imgs.append(plot_img)
            # IPython.display.clear_output(wait=True)
            # IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    # IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14, 4))
    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss


def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(10, 10))
        plt.imshow(best_img)
        plt.title('Output Image')
        plt.show()


def main(content_path, style_path):
    plt.figure(figsize=(10, 10))
    content = load_img(content_path).astype('uint8')
    style = load_img(style_path).astype('uint8')

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.show()

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']

    best, best_loss = run_style_transfer(content_path, style_path,
                                         style_layers, content_layers, num_iterations=1000)
    Image.fromarray(best)
    show_results(best, content_path, style_path)


if __name__ == "__main__":
    mpl.rcParams['figure.figsize'] = (10, 10)
    mpl.rcParams['axes.grid'] = False

    parser = argparse.ArgumentParser(description='Generate styled art through AI')
    parser.add_argument('content_path', type=str, help='image path to be styled')
    parser.add_argument('style_path', type=str, help='image path of styled image or artist style')
    args = parser.parse_args()

    if args.style_path in ASSOCIATION:
        main(args.content_path, ASSOCIATION[args.style_path])
    else:
        main(args.content_path, args.style_path)
