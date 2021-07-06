import tensorflow as tf
import keras.backend as K

def _resize_by_axis_trilinear(images, size_0, size_1, ax):
    """
    Resize image bilinearly to [size_0, size_1] except axis ax.
        :param image: a tensor 4-D with shape 
                        [batch, d0, d1, d2, channels]
        :param size_0: size 0
        :param size_1: size 1
        :param ax: axis to exclude from the interpolation
    """
    resized_list = []

    if (images.shape[ax]  == None):
        return images

    # unstack the image in 2d cases
    unstack_list = tf.unstack(images, axis = ax)
    for i in unstack_list:
        # resize bilinearly
        resized_list.append(tf.image.resize_bilinear(i, [size_0, size_1]))
    stack_img = tf.stack(resized_list, axis=ax)

    return stack_img

# def resize_by_axis(image, dim_1, dim_2, ax, is_grayscale):

#     resized_list = []

#     if is_grayscale:
#         unstack_img_depth_list = [tf.expand_dims(x,2) for x in tf.unstack(image, axis = ax)]
#         for i in unstack_img_depth_list:
#             resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
#         stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
#         print(stack_img.get_shape())

#     else:
#         unstack_img_depth_list = tf.unstack(image, axis = ax)
#         for i in unstack_img_depth_list:
#             resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
#         stack_img = tf.stack(resized_list, axis=ax)

#     return stack_img

def resize_trilinear(images, size):
    """
    Resize images to size using trilinear interpolation.
        :param images: A tensor 5-D with shape 
                        [batch, d0, d1, d2, channels]
        :param size: A 1-D int32 Tensor of 3 elements: new_d0, new_d1,
                        new_d2. The new size for the images.
    """
    assert size.shape[0] == 3
    resized = _resize_by_axis_trilinear(images, size[0], size[1], 2)
    resized = _resize_by_axis_trilinear(resized, size[0], size[2], 1)
    return resized


# jumping some lines...


def resize_multilinear_tf(images, size):
    """
    Resize images to size using multilinear interpolation.
        :param images: A tensor with shape 
                        [batch, d0, ..., dn, channels]
        :param size: A 1-D int32 Tensor. The new size for the images.
    """
    if size.shape[0] == 2:
        resized = tf.image.resize_bilinear(images, size)
    elif size.shape[0] == 3:
        resized = resize_trilinear(images, size)
    else:
        raise NotImplementedError('resize_multilinear_tf: dimensions \
                                    higuer than 3 are not supported.')
    return resized

def resize_3d(input_tensor, size):
    b_size, x_size, y_size, z_size, c_size = input_tensor.shape.as_list()
    x_size_new, y_size_new, z_size_new = size

    # resize y-z
    squeeze_b_x = K.reshape(
        input_tensor, (-1, y_size, z_size, c_size))
    resize_b_x = tf.image.resize_bilinear(
        squeeze_b_x, [y_size_new, z_size_new])
    resume_b_x = tf.reshape(
        resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

    # resize x
    #   first reorient
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    #   squeeze and 2d resize
    squeeze_b_z = tf.reshape(
        reoriented, [b_size, y_size_new, x_size, c_size])
    resize_b_z = tf.image.resize_bilinear(
        squeeze_b_z, [y_size_new, x_size_new])
    resume_b_z = tf.reshape(
        resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

    output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output_tensor