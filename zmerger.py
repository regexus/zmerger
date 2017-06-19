import cv2
import numpy as np
from time import time
from multiprocessing import Pool

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
mod = SourceModule("""
_global_ void blend_pixel_stacks(float* pixel_stacks_as_1d_array, int* image_count, float* output_image)
{
    int STACK_ITEM_DIM = 5; // RGBA + Mode
    int img_count = image_count[0];
    int stack_coord = (blockIdx.y*gridDim.x + blockIdx.x)*img_count*STACK_ITEM_DIM;
    int new_image_pixel_block_coord = (blockIdx.y*gridDim.x + blockIdx.x)*4;

    if (img_count>1)
    {
        // Starting from the second element in the stack (from the deeper side)
        // The deepest element in the array was copied to the output image (as a starting point).
        for (int img_ind=img_count-2; img_ind>-1; --img_ind)
        {
            float src_alpha = output_image[new_image_pixel_block_coord + 3]; // alpha value
            float dst_alpha = pixel_stacks_as_1d_array[stack_coord+img_ind*STACK_ITEM_DIM+3]; // alpha value

            // Early termination for black alpha
            if (dst_alpha==0.0) continue;

            float out_alpha = dst_alpha + src_alpha*(1-dst_alpha);
            float f; // Compositing mode function value stored here

            float mode = pixel_stacks_as_1d_array[stack_coord+img_ind*STACK_ITEM_DIM+4]; // mode value

            float src_rgb = output_image[new_image_pixel_block_coord + threadIdx.x]; // rgb value
            float dst_rgb = pixel_stacks_as_1d_array[stack_coord+ img_ind*STACK_ITEM_DIM+threadIdx.x]; // rgb value

            // Normal mode
            if (mode==0.0)
                f = dst_rgb;

            // Multiply mode
            else if (mode==1.0)
                f =  src_rgb * dst_rgb;

            // Screen mode
            else if (mode==2.0)
                f =  src_rgb + dst_rgb - src_rgb * dst_rgb;

            float out_rgb = (1-dst_alpha/out_alpha)*src_rgb + (dst_alpha/out_alpha)*((1-src_alpha)*dst_rgb + src_alpha*f);

            output_image[new_image_pixel_block_coord+threadIdx.x] = out_rgb;
            output_image[new_image_pixel_block_coord+3] = out_alpha;
        }
    }
}
""")

np.set_printoptions(linewidth=150, precision=4, suppress=True)

def normalize_to_float32(array):

    max_value = np.iinfo(array.dtype).max
    array = array.astype(np.float32, copy=False)
    array /= max_value

    return array

def get_rgbazm_and_res(image_data):

    # Read the RGBA image
    rgba = cv2.imread(image_data["I"], cv2.IMREAD_UNCHANGED)
    
    # Saving the image resolution. Notice the order, height comes first!
    image_heigth = rgba.shape[0]
    image_width = rgba.shape[1]

    # Normalize and resolve one dimension, 
    # to make the image array two dimensional
    rgba = normalize_to_float32(rgba)
    rgba = rgba.reshape(image_width*image_heigth, rgba.shape[2])

    # Read the Z image
    z = cv2.imread(image_data["Z"], cv2.IMREAD_GRAYSCALE)

    # Checking the z-pass resolution.
    assert image_heigth == z.shape[0], "Resolution errror! Z-Pass does not match to RGBA."
    assert image_width == z.shape[1], "Resolution errror! Z-Pass does not match to RGBA."

    z = normalize_to_float32(z)
    z = z.reshape(image_width*image_heigth, 1)

    # Create an array from mode values
    m = np.full_like(z, image_data["M"])
   
    rgbazm = np.hstack([rgba, z, m]).reshape(image_heigth*image_width, 1, 6)

    return rgbazm, image_width, image_heigth

def generate_pixel_stacks(images_data):
    """
    'images_data' is a list of dictionaties with image and mode information.
        It has following form:
        [
            {
                "I" : "full path to the image", 
                "Z" : "full path to the corresponding z-pass image", 
                "M" : "blending mode"
            },
            ...
            {
                "I" : "full path to the image", 
                "Z" : "full path to the corresponding z-pass image", 
                "M" : "blending mode"
            }
        ]
    """
    rgbazm_datas = []
    
    first_image_width = None
    first_image_heigth = None 

    for image_data in images_data:

        rgbazm, image_width, image_heigth = get_rgbazm_and_res(image_data)

        # Checking the resolution consistance
        if first_image_heigth is None:
            first_image_heigth = image_heigth
            first_image_width = image_width
        assert first_image_heigth == image_heigth, "Image resolution error!"
        assert first_image_width == image_width, "Image resolution error!"
            
        rgbazm_datas.append(rgbazm)

    # Generate the pixel stacks
    pixel_stacks = np.column_stack(rgbazm_datas)
    return pixel_stacks, image_width, image_heigth

def sort_by_z(pixel_stacks):

    sorting_order = pixel_stacks[:, :, 4].argsort()
    rows = np.arange(pixel_stacks.shape[0]).reshape(pixel_stacks.shape[0], 1)
    columns = sorting_order
    # Sort and get rid of z-component
    return pixel_stacks[rows, columns][:, :, [0, 1, 2, 3, 5]]

def blend_pixels(pixel_info_src, pixel_info_dst):
    
    dst_alpha = pixel_info_dst[3]
    src_alpha = pixel_info_src[3]
    out_alpha = dst_alpha + src_alpha*(1-dst_alpha)
    src_rgb = pixel_info_src[:3]
    dst_rgb = pixel_info_dst[:3]

    # Normal mode
    if pixel_info_dst[-1]==0.0:
        f = dst_rgb

    # Multiply mode
    elif pixel_info_dst[-1]==1.0:
        f =  src_rgb * dst_rgb

    # Screen mode
    elif pixel_info_dst[-1]==2.0:
        f =  src_rgb + dst_rgb - src_rgb * dst_rgb

    out_rgb = (1-dst_alpha/out_alpha)*src_rgb + (dst_alpha/out_alpha)*((1-src_alpha)*dst_rgb + src_alpha*f)

    pixel_info_dst[:3] = out_rgb
    pixel_info_dst[3] = out_alpha

def compute_pixel_stack(pixel_stacks):
    """ A pixel stack is an array of arrays like this:
        [[R, G, B, A, Z, M], ... , [R, G, B, A, Z, M]]
        (M means Mode).
    """

    # Apply alpha blending
    # 0 - normal mode


    # Find all non-transparent pixels in normal mode
    # print non_transparent_pixels

    for pixel_coord, pixel_stack in enumerate(pixel_stacks):

        if not pixel_stack[:, 3].any():
            # print "Full transparent"
            continue

        first_relevant_index = pixel_stack.shape[0]-1
        for i, pixel_info in enumerate(pixel_stack):
            if (not pixel_stack[i+1:, 3].any()) or (pixel_info[3] == 1.0 and pixel_info[-1]==0.0):
                first_relevant_index = i
                break
        if first_relevant_index>0:
            for j in range(first_relevant_index, 0, -1):
                blend_pixels(pixel_stack[j], pixel_stack[j-1])
                # if j-1 == 0:
                    # print "Yo", first_relevant_index, pixel_coord, "\n", pixel_stack
                    # return


    # first_relevant_index = non_transparent_pixels[0]
    # if first_relevant_index>0:
    #     for i in xrange(first_relevant_index, reversed=True)
    #         blend_pixels(pixel_info_src, pixel_info_dst, mode)

    # pass
