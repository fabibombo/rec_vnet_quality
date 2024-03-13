import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve

### INHOMOGEINITIES ###
def simulate_inhomogeneities(slice, max_artifact_px_radius=64, num_inhomogeneities=1):

    h, w = slice.shape

    # normalization to pick artifact center
    min_intensity = np.min(slice)
    max_intensity = np.max(slice)
    norm_slice = (slice - min_intensity) / (max_intensity - min_intensity)

    # Define the brightness factor
    out_factor = 1
    mask = np.full_like(slice, out_factor)

    for _ in range(num_inhomogeneities):
        pick_center = True
        region = 16
        x_center, y_center = region, region
        while (pick_center or np.mean(norm_slice[y_center-region:y_center+region, x_center-region:x_center+region]) < 0.1):
            x_center = np.random.randint(region, w - region)
            y_center = np.random.randint(region, h - region)
            pick_center = False
    
        # Ellipse area of inhomogeneity
        brightness_factor = np.random.uniform(low=1.5, high=6, size=1)

        if np.random.rand() < 0.5:
            brightness_factor = 1/brightness_factor
        
        a, b = np.random.uniform(low=0.25, high=2, size=2)
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
    
        distance = np.sqrt(((x - x_center) / a) ** 2 + ((y - y_center) / b) ** 2)
    
        ellipse_mask = distance <= max_artifact_px_radius
        
        mask[ellipse_mask] = brightness_factor

    smoothed_mask = gaussian_filter(mask, sigma=8)

    mod_slice = slice * smoothed_mask
    
    return mod_slice


### XY SHIFTS ###
def simulate_motion_artifacts(slice, shift_ratio=0.15):
    # Get volume dimensions
    height, width = slice.shape

    max_shift = int(height*shift_ratio)
    
    # Randomly generate shifts along each axis
    shift_x = np.random.randint(-max_shift, max_shift)
    shift_y = np.random.randint(-max_shift, max_shift)
    #shift_z = random.randint(-max_shift, max_shift)
    
    # Apply shifts to the MRI volume
    shifted_slice = np.roll(slice, shift_x, axis=0)
    shifted_slice = np.roll(slice, shift_y, axis=1)
    #shifted_volume = np.roll(shifted_volume, shift_z, axis=2)
    
    return shifted_slice


### BLURRING GHOSTING IN MOTION ###
def simulate_blurring_ghosting(slice, blur_strength=1):
    """
    Simulate blurring or ghosting effects by convolving the MRI volume with a blurring kernel.
    
    Parameters:
        slice (ndarray): 2D numpy array representing the MRI frame.
        kernel_size (int): Size of the blurring kernel.
        blur_strength (float): Strength of the blurring effect.
    
    Returns:
        ndarray: MRI frame with simulated blurring or ghosting effects.
        
    """

    blur_strength = np.random.uniform(low=1, high=3)
    kx, ky = np.random.randint(low=1.5, high=5, size=2)

    # Generate blurring kernel
    kernel = np.ones((kx, ky)) / (kx*ky)
    
    # Apply convolution to simulate blurring or ghosting
    mod_slice = convolve(slice, kernel)
    
    # Add blurred volume to original volume with reduced intensity
    mod_slice = blur_strength * mod_slice + (1 - blur_strength) * (slice)
    
    return mod_slice

### ALISING ###
# To simulate undersampling
def simulate_aliasing(mri_image, undersample_factor=2, interpolation='nearest'):
    """
    Simulate aliasing effects by downsampling and then upsampling the MRI volume.
    
    Parameters:
        mri_image (ndarray): 2D numpy array representing the MRI frame.
        undersample_factor (int): Factor by which to downscale the volume.
        interpolation (str): Interpolation method to use during upsampling.
            Options: 'nearest', 'linear', 'cubic'
    
    Returns:
        ndarray: MRI frame with simulated aliasing effects.
    """
    undersample_factor = np.random.uniform(low=1, high=3)
    undersample_factor = np.round(undersample_factor, decimals=1)
    
    # Downsample the volume
    downsampled_volume = zoom(mri_image, 1/undersample_factor, order=1)
    
    # Upsample the downsampled volume to original size
    upsampled_volume = zoom(downsampled_volume, undersample_factor, order=1)

    return upsampled_volume


### GRADIENT NONLINEARITY ###
    
def simulate_gradient_nonlinearity(slice):
    """
    Simulate gradient nonlinearity effects by applying spatially varying deformations to the MRI volume.
    
    Parameters:
        slice (ndarray): 2D numpy array representing the MRI frame.
    
    Returns:
        ndarray: MRI frame with simulated gradient nonlinearity effects.
    """

    # it could be done to 3d if we want to
    
    # Generate distorted coordinates
    amp_distortion_x = np.random.uniform(low=0.1, high=2)
    amp_distortion_y = np.random.uniform(low=0.1, high=2)
    distortion_factor_x = np.random.uniform(low=0.1, high=1.25)
    distortion_factor_y = np.random.uniform(low=0.1, high=1.25)
    
    x, y = np.indices(slice.shape, dtype=float)
    distorted_x = x + distortion_factor_x * np.sin(amp_distortion_x*y)
    distorted_y = y + distortion_factor_y * np.sin(amp_distortion_y*x)
    
    # Interpolate the distorted volume
    distorted_slice = np.zeros_like(slice)

    interpolator = RegularGridInterpolator((np.arange(slice.shape[0]),
                                            np.arange(slice.shape[1])),
                                           slice,
                                           bounds_error=False,
                                           fill_value=0)
    distorted_slice = interpolator((distorted_x, distorted_y))

    return distorted_slice

### PARTIAL VOLUME EFFECTS ###
def simulate_partial_volume_effect(img, sigma=1.0):
    """
    Simulate partial volume effects by convolving the MRI volume with a Gaussian point spread function (PSF).
    
    Parameters:
        image (ndarray): numpy array
        sigma (float): Standard deviation of the Gaussian PSF, controlling the amount of blurring.
    
    Returns:
        ndarray: MRI volume with simulated partial volume effects.
    """
    sigma = np.random.uniform(low=0.5, high=2)
    
    blurred_img = gaussian_filter(img, sigma=sigma)
    
    return blurred_img


### 3D GHOSTING ###
# def simulate_ghosting_3d(mri_volume, ghosting_factor=0.12, p=0.33):
#     """
#     Simulate ghosting artifacts by introducing duplicated images shifted along the phase-encoding direction.
    
#     Parameters:
#         mri_volume (ndarray): 3D numpy array representing the MRI volume.
#         ghosting_factor (float): Factor controlling the strength of the ghosting artifacts.
    
#     Returns:
#         ndarray: MRI volume with simulated ghosting artifacts.
#     """
#     ghosting_shifts = np.random.uniform(low=0.1, high=0.2)
#     ghosting_intensity = np.random.uniform(low=0.05, high=0.5)
    
#     # Determine the number of shifted copies
#     num_shifts = int(ghosting_shifts * mri_volume.shape[2])
    
#     # Generate shifted copies
#     ghosted_volume = np.zeros_like(mri_volume)
#     for shift in range(1, num_shifts + 1):
#         ghosted_volume[:,:,:-shift] += ghosting_intensity*mri_volume[:,:,shift:]
#     # Combine with the original volume
#     ghosted_volume += mri_volume

#     for idx in range(mri_volume.shape[2]):
#         if np.random.uniform() > p:
#             ghosted_volume[:,:,idx] = mri_volume[:,:,idx]
    
#     return ghosted_volume

### 2D GHOSTING ###
def simulate_axis_ghosting(mri_volume, grid_value=13, ratio=2):
    """
    Simulate vertical ghosting artifact by summing one cell to the rest of the cells in the same column.
    
    Parameters:
        mri_volume (ndarray): 3D numpy array representing the MRI volume.
    
    Returns:
        ndarray: MRI volume with simulated vertical ghosting artifact.
    """
    grid_value_x = np.random.randint(2, 16)
    grid_value_y = np.random.randint(2, 16)
    intensity_factor = np.random.uniform(low=0.25, high=0.75)
    blur_factor = np.random.uniform(low=0.1, high=2)
    
    # Get the dimensions of the MRI volume
    width, height = mri_volume.shape
    mri_volume_tmp = mri_volume.copy()
    
    # Define the grid size
    grid_width = width//grid_value_x
    grid_height = height//grid_value_y

    # Randomly pick the i,j coordinate for the chosen cell
    selected_i = np.random.randint(grid_value_x-1)
    selected_j = np.random.randint(grid_value_y-1)
    
    cell = mri_volume[int(selected_i*grid_width):int((selected_i+1)*grid_width),
                        int(selected_j*grid_height):int((selected_j+1)*grid_height)]

    cell = gaussian_filter(cell, sigma=blur_factor)
    
    if np.random.uniform() < 0.5:
        for idx in range(grid_value_y):
            if idx != selected_j:
                mri_volume_tmp[int(selected_i*grid_width):int((selected_i+1)*grid_width),
                                int(idx*grid_height):int((idx+1)*grid_height)] *= 1-intensity_factor
                mri_volume_tmp[int(selected_i*grid_width):int((selected_i+1)*grid_width),
                                int(idx*grid_height):int((idx+1)*grid_height)] += intensity_factor*cell
    else:
        for idx in range(grid_value_x):
            if idx != selected_i:
                mri_volume_tmp[int(idx*grid_width):int((idx+1)*grid_width),
                                int(selected_j*grid_height):int((selected_j+1)*grid_height)] *= 1-intensity_factor
                mri_volume_tmp[int(idx*grid_width):int((idx+1)*grid_width),
                                int(selected_j*grid_height):int((selected_j+1)*grid_height)] += intensity_factor*cell

    return mri_volume_tmp

### NOISE ###
def add_gaussian_noise(image, mean_factor=0.5, std_factor=0.1):
    mean_factor = np.random.uniform(low=0.1, high=1)
    std_factor = np.random.uniform(low=0.1, high=0.6)
    
    # Generate Gaussian noise with the same shape as the MRI volume
    noise = np.random.normal(image.mean(), std_factor*image.std(), image.shape)
    
    # Add the noise to the MRI volume
    noisy_img = image + noise

    return noisy_img
