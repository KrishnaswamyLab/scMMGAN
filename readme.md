This is the code for implementing the neural network model Single-Cell Multi-Modal GAN (scMMGAN). The model maps between two domains as inferred from two discrete datasets. The structure of the script is as follows:

# Loading data
The `x1` and `x2` python variables must hold the input datasets ready for feeding into the neural network. The best results are achieved with preprocessing that ensures each input column is mostly in the interval [-10, 10]. Further preprocessing may be necessary depending on the specific data properties.

# Hyperparameters
`TRAINING_STEPS`: can be chosen based on how output is evolving. When the generated output is stable across training iterations, further training is unnecessary.
`batch_size` and `nfilt`: larger batch sizes and more free parameters generally produce better results, so these numbers can be increased based on computational resources and time.
`learning_rate`: changing learning rates has rarely been effective, observationally
`lambda_cycle`: increase this if the reconstruction is retaining too little global structure from the original data
`lambda_correspondence`: increase this if the aligned mapping is retaining too little of the data geometry (e.g. nearby points in the input space are not nearby in the aligned space)
`add_noise`: used to measure uncertainty if wanted
`use_bn`: layer-based batch normalization is generally used in the most effective model architectures, but sometimes turning it off is helpful

# Function definitions and Tensorflow graph construction
Running lines through the `sess.run` call in line 421 prepare the code but do not start any training yet.

# Training
Each step in the `while` loop executes one training iteration. The generator is first optimized with `train_op_G` and then the discriminator is optimized with `train_op_D`. Helpful loss calculations for santity checks are done every 100 steps via the `training_counter` variable. Additional checks like plotting can be done intermittently in this section, as well.

# Getting output
The variable `output_fake2` contains the generated data of mapping x1 into the second domain and the variable `output_fake` contains the generated data of mapping x2 into the first domain. Any preprocessing done to morph the data into the desired structure (e.g. scaling) could be inverted here to return the data to its original form.

