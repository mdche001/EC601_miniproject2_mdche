# EC601_miniproject2_mdche
    Mini project for EC 601 in BU

The repository contains two trained model for miniproject2: Keras model, CIFAR-10-like model;<br>  
    --Keras model which is pretrained model is directly called by car.py <br>
    --CIFAR-10-like model which is similar to CIFAR-10 model posted on the Google webiste contains following files: <br>
    
    1. python script:
      ·input_data.py: Reading images from the miniprojcet2/data/train and miniprojcet2/data/test as the input of training and testing.
      ·model.py: Defining the neural network for the training 
      ·training.py: Training the model which is in model.py and test is also in this script.
      ·frozen.py: Freezing the trained model in miniproject2/logs/train, and creating a .pb file for the customer use.
      
    2. model body:
      ·which is stored in my loscalhost, coz the pipe always broke when I upload the model files to github. 
       The error is as follows:     
![image](https://github.com/mdche001/EC601_miniproject2_mdche/blob/master/images/model_upload_error.JPG)
    
    3. Dataset：
       ·The dataset used in this project is downloaded from kaggle:https://www.kaggle.com/vikashvverma/flowers-classification/data
        It contains five classes related to flowers and two of them is used for my training and test.
        The images is about 1'500 in total.
    
    4. Comparison between two model:
       ·The test data comes from miniprojcet2/data/test, the test result of CIFAR-10-like model is as follows:
![image](https://github.com/mdche001/EC601_miniproject2_mdche/blob/master/images/testing.JPG) 
    
       ·Also, the test data is from same directory and the reseult is shown in the figure: 
    
    
