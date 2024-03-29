# fisheye_tools
Tools to process fisheye images

## Crop and resize

Convolutional Neural Network models often require square images of low resolution, varying form 28x28 to 300x300 pixels.

The 'eye_crop_resize.py' script takes as input fisheye images of any size and resolution.
It produces for each image a reduced, square image containing just the eye of the image.

Crop and resize one image using default parameters:
```
python eye_crop_resize.py /PATH/TO/IMAGE
```

Crop and resize a whole directory of images using default parameters:
```
python eye_crop_resize.py /PATH/TO/IMAGE_DIRECTORY
```

The script takes following optional arguments:
```
  -h, --help            show help message and exit
  -o IMAGE_OUTPUT_PATH, --image_output_path IMAGE_OUTPUT_PATH
                        /path/to/output_image_dir
  -s IMAGE_OUTPUT_SIZE, --image_output_size IMAGE_OUTPUT_SIZE
                        Size of output image
  -r EYE_REDUCTION, --eye_reduction EYE_REDUCTION
                        Eye size reduction factor
  -d, --display         Display image
```

>Note: If no output path is specified the resized output images are stored in the same directory as the original images.

>Note: The eye size reduction factor (default: 0.9) is useful to remove the outer border of the eye. Just set eye reduction to 1.0 if you do not want to remove the outer border. 

>Note: Press spacebar to close currently displayed image en proceed to next image.




