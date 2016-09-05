# **Fingerprint**

## Explaination of Different Jupyter Notebook Files
- **Digit**: 

To test the siamese network structure works, a simpler network was created to execute MNIST digit data. The network consists of only one side of the original siamese net and was run to test whether it could successfually learn anything.

The results are good as the network can produce a test accuracy of at least 96% on test set (previously unseen) with only 1000 training iterations (tensorflow's original example needs at least 15k iterations to achieve such results)

- **Updated_Siamese_net**: 

First a few global variables are defined: 

  1. `raw_only`: a flag marks if we are only using the centered and cropped raw image;
  
  2. `image_size`: the size of input image;
  
  3. `num_layer`: the number of layers `raw_only` flag is `False`. 
  
Then 

## Function Documentation

`load_pairs_from_preprocessed(match_path, mismatch_path)`

Returns `images_match, images_mismatch`, two dictionaies of images with `num_layer` layers and key'ed like `1_0`, `1_1`.

`generate_batch_pairs_from_preprocessed(images_match, images_mismatch, num, image_size)`

Returns `[x, x_p, y]`, similar to the MNIST digit batch generator provided by tensorflow. Output is a list, with the first two items are an matrix of `num` rows and `image_size * image_size * num_layer` columns; the third item is a matrix of `num` rows and 1 columns, with 1 if the two finger prints are a match and 0 if they are not a match. 

`suffle_all(x, x_p, y)`:
Function to shuffle a batch
