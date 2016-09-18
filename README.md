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

- **Pearson_NoTrain**: 

From Isabelle and Cedric. This extract the Pearson Correlation Coefficient from the 6 layers, 'Original image', 'Frequency map', 'Orientation map', 'Abs orient. map', 'Variance map', 'Gabor', and achieved a AUC of 0.86 for combination of the 6 layers. 

## Function Documentation

### `importFunctions` imported as `iF`

`load_pairs_from_preprocessed(match_path, mismatch_path, image_size, num_layer, small_set=True)`

If 'small_set' is 'True', then returns `images_match, images_mismatch`, two dictionaies of images with `num_layer` layers and key'ed like `1_0`, `1_1`, with data generated from Cedric's smaller 94-pairs-each data set.

If 'small_set' is 'False', then returns with the larger dataset. Notice that the IDs are not continuous, due to the fact that some pairs have low image quality and they do not qualify for training.

`generate_batch_pairs_from_preprocessed(images_match, images_mismatch, num, image_size)`

Returns `[x, x_p, y]`, similar to the MNIST digit batch generator provided by tensorflow. Output is a list, with the first two items are an matrix of `num` rows and `image_size * image_size * num_layer` columns; the third item is a matrix of `num` rows and 1 columns, with 1 if the two finger prints are a match and 0 if they are not a match. 

`suffle_all(x, x_p, y)`:
Function to shuffle a batch

`show_ROC(actual, predictions, title)`

### In `Updated_Siamese_net`

`load_pairs_from_preprocessed(match_path, mismatch_path, small_set=True)`
