# Introduction to Generative AI practical exercise

## Task Instructions

The task will be to implement Latent Diffusion (or Latent Flow).
The idea is to use the EMNIST dataset, which is more challenging than MNIST because it contains both digits and letters, and train a VAE encoder-decoder with a much smaller latent space. Then, you will perform Deterministic Diffusion (or, more generally, a Flow) in this latent space and finally map the generated samples back to the data space using the decoder.
You should organize yourselves into groups of up to 5 people. Please declare your groups here:
https://docs.google.com/spreadsheets/d/1U0kNotSrl2bhP116PWrqtg9iUfemJmjaWeYWQJ5yfCs/edit?usp=sharing
The entire project should be implemented in Google Colab with pytorch. You may use the free GPU resources provided by Google Colab, but I recommend also purchasing GPU Units. Around 100 GPU Units should be more than enough for the project. The cost is not very high (approximately R$ 60), which should be reasonable for a group of 5 people.
You can obviously use generative AI to help you. But you should know what are you doing, because I will ask questions.

## Deliverables

The deliverables are:
-The trained Encoder, Decoder, and Denoiser/Flow models, saved as checkpoint files (.ckpt), which should be sent to me for evaluation. This should be delivered by July 1st.
-A presentation at the end of the semester. The presentations will take place online through Google Meet on July 2 and 3, at a time slot to be agreed upon between the group and me. Each presentation should last 25 minutes. The presentation may be given either in Portuguese or English.


- In the presentation, you should discuss:
    - In the VAE:
        - The chosen architecture;
        - Training details (learning rate, epochs, batch size);
        - The chosen likelihood model ( f_{X|Z}(x,z) ) (Gaussian, Bernoulli, Laplace, etc.);
        - Generation and reconstruction examples;
        - Snapshots of 2D marginals of the latent space to illustrate the quality of the learned representation.
        - Plots of the training error as a function of the number of steps.
    - In the Diffusion/Flow model:
        - The chosen architecture;
        - Training details learning rate, epochs, batch size;
        - The chosen scheduler;
        - The chosen sampler.
        - Plots of the training error as a function of the number of steps.
        - Snapshots of the Generation