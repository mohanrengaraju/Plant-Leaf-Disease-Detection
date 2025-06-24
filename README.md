  PLANT LEAF DETECTION SYSTEM
  
##Problem Statement:
        The goal of this project is to develop an image classification model capable of identifying common diseases in plant leaves. Early detection of plant diseases is crucial for effective treatment and preventing widespread crop loss.
    
## 🛠️ Technologies Used

PIL (Pillow): For opening and displaying image files.

IPython.display: For displaying images within the notebook.

tensorflow\.keras: The deep learning framework used to build and train the convolutional neural network (CNN) model.

 ImageDataGenerator: For data augmentation and preparing image data for the model.
 
 Sequential: To define the linear stack of layers for the model.
 
 Conv2D, MaxPooling2D, Flatten, Dense: Layers used in the CNN architecture.
 
 numpy: For numerical operations, particularly for handling image data arrays and model predictions.
 
 matplotlib.pyplot: For plotting the training and validation accuracy.
 
 seaborn: For enhancing the appearance of the plots.

## 📊 Model Metrics and Results

The model was trained for **5 epochs**, using accuracy** and **categorical crossentropy loss** as evaluation metrics.

Training Accuracy: \~91%
Validation Accuracy: \~83%
Observation: The loss steadily decreased over time, indicating effective learning. However, the gap between training and validation accuracy suggests **slight overfitting**, which could be improved with:

  * Additional data
  * Regularization techniques (e.g., dropout, L2 regularization)
  * Early stopping
![accuracy](https://github.com/user-attachments/assets/27b75542-fb49-47aa-8dd7-59da3c774033)

##Sample Output:
      The prediction on a sample image from the test set resulted in the following probability distribution across the classes:
          
          Healthy: ~27.88%
          Powdery: ~0.01%
          Rust: ~72.11%
          Based on these probabilities, the model predicted the image to be Rust, which matches the directory the image was taken from (Test/Test/Rust/).
   ![output](https://github.com/user-attachments/assets/91d69a06-4ee6-47e6-8ecf-34337e652f19)

##FOR DATASET:
Kaggle API: Used to download the dataset.

 ![plant leaf teat and train](https://github.com/user-attachments/assets/3c903f50-07e4-4387-afb2-6af65e97fa56)

