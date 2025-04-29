# Denoising Models 

## Goal: 

Our goals it to denoise frames before running postural tracking in order to reduce noise. This step is beneficial to the overall pipeline as it reduces the number of errors during the prediction stage. 

## Data Description:

To train our models throuh the denoing step, we have decided to start from the keyframe extracted. The dataset of key frames is composed of 500 key frames per video variable shape depending on the video view (left, right:220x624, top, bottom: 220x608 or central:608x624). 

To ensure consistency across different model's architectures we have decided to convert key-frames into patches of the same size. Patches were formed in the following way: given a Tensor of frames of shape (frames, h, w) a patch size is selected (e.g., 64). Following to ensure that the input tensor is divisble by the patch we add padding to each dimension using the module operator (pad = dim - (h%dim) if ! 0 else 0), the padding is applied symmetrically to the image. 

Once the number of patches of each dimension is determined, we have the final shape: frames, number of patches, dim, dim which will rearranged in: (frames*number of patches), dim, dim.

The input tensor is normalised to contain values only between 0 and 1. 

In this way we have for a video of the lateral view the following input tensor: 20k, 64, 64


## Architectures Considered:

### AutoEncoders:

The natural choice to denoise data are autoencoders. Thise models takes an input $$x$$ (i.g., image) then they process it to create a latent space, with a smaller dimensions compared to the original input, during this process the encoder learns how to represent the most important features of the input. Then the decoder does the inverse operation trying to reconstruct the input starting from the $z$ point the latent space. The loss is the difference between the input and the reconstructed output. 

We consider different kinds of autoencoders:

1. Denoising autoencoder: 

The model has been built using the PyTorch library, it has two hyperparameters the hidden channel dimension and the latent space size, they determined the size of the latent space of the autoencder. 

The autoenconder has two main modules: an encoder, that has to transform the input image into a datapoint in the latent space; and the decoder, that starts from a point in the latent space and it has to reconstruct the input from that. 

The encoder contains: two convolutional blocks (each of them formed by a 2d convolution, ReLU as activation function and the first one has a Batch normalzation module), then the input is flattent and passed to two linear blocks (Linear layer, the first one has ReLU as activation function) that will map the input into the latent space.  

The decoder on the other hand has the opposite architecture of the encoder. It starts with two linear layers, then the input is mapped into image shape to be passed to two transposed convolutions blocks that will produce a final image of the same shape of the orignal input. 

During the forward pass we pass an input image to the encoder that produces a point in the latent space which is passed to the decoder which will produce an output image. 

2. Sparse autoencoder: 

3. Variational Autoencder 

4. UNet: encoder-decoder with skip connections 

The model consists of two main componetns:

- Noise estimator Network: simple 5 fully convolutional layers=> it servers to generate a map of the noise in the img
- UNet: series of convolutions downsampling and upsampling with skip connections (input is img + noise estimation)








## Training Regime:

 The input tensor consists of grayscale video frames shaped as (frames, height, width), which are first normalized to the [0, 1] range by dividing by 255. These frames are then split into non-overlapping patches of size dim x dim (e.g. 64x64) using a custom patching function. The resulting dataset consists of all extracted patches reshaped into the format (total_patches, 1, dim, dim), which is then split into training and testing sets with a configurable ratio (default 80/20).

The model is a standard convolutional autoencoder consisting of an encoder and decoder. The encoder applies two 2D convolutional layers followed by ReLU activations and batch normalization, then flattens the output and passes it through two linear layers to produce a latent representation of size latent_dim_size. The decoder mirrors this structure: it starts from the latent space, passes through linear layers, reshapes the tensor to the original conv shape, and reconstructs the image with transposed convolution layers.

Training is performed using the Adam optimizer with configurable learning rate and beta values. The loss function is MSELoss, which computes the mean squared error between the reconstructed and original normalized patches. The model is trained on an MPS device (Apple GPU), and batches are loaded with a configurable batch size (default 32).

To monitor model performance during training, a holdout set of patches corresponding to one full frame is reserved. These are passed through the model at regular intervals (log_every_n_steps), and the reconstructed patches are assembled back into a full frame using the inverse of the patching process. Both the original and reconstructed images are saved or logged (optionally via Weights & Biases) for qualitative inspection.

## RIDNet: Real Image Denoising with Feature Attention

There are three main modules:

1. Feature extraction

It's a simple convolutional layer to extract features. Given the noisy input x it produces $f_0 = M_{fl}(x)$


2. feature learning residual (residual modu

M_{fl} takes the output of the convolutional layer and produces f_r 

3. Reconstruction 
 it is a simple convolitions to reconstruct the original image. 

**Loss**: l1 or MAE over batch dim. 

### Feature Learning Residual on the Residual 




## Pyramid Real image Denoising network: PRIDNet

The innovative components are:

- Channel attention: it extracts noise features by recalibrating the channel importantce (as different channels carry different noise information) = not good for the project as we deal with gray scale images
- Multi-scale feature extraction: pyramid structure, each branch pays attention to one-scale features. 

- Feature self-adaptive fusion: each channel represents one-scale features, here they fuse the different scales into a single one. 


**Network**: three stages: noise estimation, multi-scale denoising, feature fusion. 

1. Noise estimation: plain five-layer fully conv (pooling, Batch norm, ReLU after each conv). Conv2D(k=3, oc = 32). In addition, they insert an attention module to calibrate the channels features (not necessary as we have one single channle, maybe it would be cool to use patches as channels)

2. Multi-scale denoising stage: 

It is a five layer pyramid, the input feature maps are downsampled to different sizes, with the goal of capturing original, local and global information at the same time. 
Pooling kerneles are set to: 1x1, 2x2, 3x3, 4x4, 8x8, 16x16
After each pooled feature there is U-Net, total 5 unet with independent weights. At the final stage the multi-level features are upsampled by bilinear interpolation to the same size and concatenated

3. Feature Fusion Stage: 

Kernel selecting module 

The network is trained optimizing the l1 loss 



## Metrics:

1. MSE V
2. SSIM: structure similarity 
3. Impact over keypoint tracking (reprojection error?)

4. Final loss: select the best models with their loss and then compare their performances over the actual task. 
So the training regime consists of train each model with their losses, then compare them and select each model to compete in the pipeline to construct the 3D image. 

- PCA Loss, reprojection loss over one video. 
