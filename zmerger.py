from time import time
import numpy as np
import cv2

np.set_printoptions(linewidth=150, precision=4, suppress=True)

IMAGES_DATA = [
    {
        "I" : r"e:\Programming\Python\Projects\2017\ZMerger\images\img1_RGBA.png", 
        "Z" : r"e:\Programming\Python\Projects\2017\ZMerger\images\img1_Z.png", 
        "M" : 2.0
    }, 
    {
        "I" : r"e:\Programming\Python\Projects\2017\ZMerger\images\img2_RGBA.png", 
        "Z" : r"e:\Programming\Python\Projects\2017\ZMerger\images\img2_Z.png", 
        "M" : 1.0
    }, 
    {
        "I" : r"e:\Programming\Python\Projects\2017\ZMerger\images\img3_RGBA.png", 
        "Z" : r"e:\Programming\Python\Projects\2017\ZMerger\images\img3_Z.png", 
        "M" : 0.0
    },
]

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
    rgba_z_and_mode_datas = []

    image_width = None
    image_heigth = None 

    for image_data in images_data:

        # Read the RGBA image
        i = cv2.imread(image_data["I"], cv2.IMREAD_UNCHANGED)

        # Checking and saving the image resolution. Notice the order, height comes first!
        if image_heigth is not None:
            assert image_heigth == len(i), "Image resolution error!"
            assert image_width == len(i[0]), "Image resolution error!"
        image_heigth = len(i)
        image_width = len(i[0])

        # Resolve one dimension, to make the image array two dimensional 
        i = i.reshape(-1, i.shape[-1])
        # Set the range to 0..1 and convert to float
        i = i/(np.iinfo(i.dtype).max + 0.0)

        # Read the Z image
        z = cv2.imread(image_data["Z"], cv2.IMREAD_GRAYSCALE)
        z = z.reshape(i.shape[0], 1)
        # Set the range to 0..1 and convert to float
        z = z/(np.iinfo(z.dtype).max + 0.0)

        # Create an array from mode values
        m = np.full_like(z, image_data["M"])
       
        # Put all the data of one image set together
        rgba_z_and_mode_datas.append(np.hstack([i, z, m]).reshape(image_heigth*image_width, 1, 6))

    # Generate the pixel stacks
    pixel_stacks = np.column_stack(rgba_z_and_mode_datas)
    return pixel_stacks, image_width, image_heigth

def sort_by_z(pixel_stacks):

    sorting_order = pixel_stacks[:, :, 4].argsort()
    rows = np.arange(pixel_stacks.shape[0]).reshape(pixel_stacks.shape[0], 1)
    columns = sorting_order
    return pixel_stacks[rows, columns]

def blend_pixels(pixel_info_src, pixel_info_dst):
    
    # Normal mode
    if pixel_info_dst[-1]==0.0:
        pixel_info_dst[:3] = (1-pixel_info_dst[3])*pixel_info_src[:3] + pixel_info_dst[:3]*pixel_info_dst[3]

    elif pixel_info_dst[-1]==1.0:
        pixel_info_dst[:3] = pixel_info_src[:3] * pixel_info_dst[:3] *pixel_info_dst[3] + pixel_info_src[:3]*(1-pixel_info_dst[3])

    elif pixel_info_dst[-1]==2.0:
        pixel_info_dst[:3] = pixel_info_src[:3]/pixel_info_src[3] + pixel_info_dst[:3]#*pixel_info_dst[3]

    pixel_info_dst[3] = (1-pixel_info_dst[3])*pixel_info_src[3] + pixel_info_dst[3]

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

# t = time()
# pixel_stacks, image_width, image_heigth = generate_pixel_stacks(IMAGES_DATA)
# print "generate_pixel_stacks", (time()-t)

# t = time()
# pixel_stacks = sort_by_z(pixel_stacks)
# print "sort_by_z", (time()-t)

# compute_pixel_stack(pixel_stacks)


# out = pixel_stacks[:, 0, :4]
# img = out.reshape(image_heigth, image_width, 4)
# img = img*255
# cv2.imwrite(r"e:\Programming\Python\Projects\2017\ZMerger\workspace\result.png", img)