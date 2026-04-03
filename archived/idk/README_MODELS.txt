MODELS
-----------------------------------------------------------------------------------

mackeyglass_single_A:

n-dim input that skips 1 step
1-dim output

RnnCell only uses the input mapping 'Wx' and learns a unitary matrix.

- Gets results that semi-map the output, but it seems heavily dependent on the input.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

mackeyglass_single_B:

n-dim input that skips 1 step
1-dim output

Uses 2x hermitian matrices to map the state and input to the Hopf-ODE. Figured out a 
backprop gradient for the hermitian case that seemed to work. Results were good but not
great. Using a smaller featuresize and epoch size worked better than a larger feature size,
which might imply that the network isn't really learning anything.

Good results:
feature_size = 128
epoch_size = 64
num_epochs = 32

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

mackeyglass_single_C:

n-dim input that skips 1 step
1-dim output

Uses 2x hermitian matrices to map the state and input to the Hopf-ODE. Has updated hermitian
gradient that seems to work better than the 'single_B' version.

Good results:
feature_size = 128
epoch_size = 64
num_epochs = 32
