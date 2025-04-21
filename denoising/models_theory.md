# Denoising Models 

## Goal: 

Our goals it to denoise frames before running postural tracking in order to reduce noise. This step is beneficial to the overall pipeline as it reduces the number of errors during the prediction stage. 


## Architectures Considered:

### AutoEncoders:

The natural choice to denoise data are autoencoders. Thise models takes an input $$x$$ (i.g., image) then they process it to create a latent space, with a smaller dimensions compared to the original input, during this process the encoder learns how to represent the most important features of the input. Then the decoder does the inverse operation trying to reconstruct the input starting from the $z$ point the latent space. The loss is the difference between the input and the reconstructed output. 

We consider different kinds of autoencoders:

1. Denoising autoencoder: we apply some noise to the input 

2. Sparse autoencoder

3. Variational Autoencder 

4. UNet: encoder-decoder with skip connections 

### Temporal Smoothing?

We use the past and future frames to clean the current one 


### Attention Based autoencoders?

## Metrics:

1. MSE
2. SSIM: structure similarity 
3. Impact over keypoint tracking (reprojection error?)