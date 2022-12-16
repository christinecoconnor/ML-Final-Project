# ML-Final-Project
Final project for Ellie Zhang, Alex Mo, and Christine O'Connor


To run our model, you can first download the dataset from the following link: https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images or using your own images. Image must be grouped into 4 sub-folder each respresentng its label and 4 sub-folders need to store in one base folder for both testing and training data. 

You will also need to make sure your IDE has all the necessary libraries/packages installed and working properly: numpy, pandas, matplotlib, tensorflow.

After this initial setup, you will need to add the paths of the downloaded train and test datasets into the train_dir and test_dir variables, respectively. Once this step is completed, you should be able to simply run the model!

you can choose to change some parameters at the TOP of the code to fit with your data: 

  IMAGE_SIZE = [204, 176]  -> could choose the oringal size or scaled to improve efficiency orignal image size
  EPOCHS = 50              -> longer epoches result better training but also could take longer 
  SPLIT = 0.2              -> choose how many % of testing data you want to split as validation 
  COLOR_MODE = "rgb"       -> choose "rgb" for 3 color channel or "graycale" is only wants black and white images
