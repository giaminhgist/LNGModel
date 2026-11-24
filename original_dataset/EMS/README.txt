Dear Researchers,

There are three folders and one file in the released database.

Three folders are "Images", "Train_Valid", and "Test", respectively. 
1. "Images" contains the 100 images as stimuli corresponding to the four categories.
2. "Train_Valid" contains the eye movement data from 160 subjects. 
Files numbered less than 200 are from healthy controls. Files with numbers greater than or equal to 200 are from schizophrenics.
3. "Test" contains the eye movement data from 48 subjects. The test set is shuffled. The file name does not contain category information. If you want to test your model on the test set of EMS, follow the steps at https://github.com/YingjieSong1/EMS?tab=readme-ov-file#benchmark.

One file is "Train_Valid.xlsx".
1. The 160 subjects corresponding to "Train_Valid" folder were split into four sets for 4-fold cross-validation.
