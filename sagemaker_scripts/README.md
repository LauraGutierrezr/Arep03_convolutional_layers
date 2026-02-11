### Deployment in Amazon SageMaker

The convolutional neural network (CNN) was trained and deployed using Amazon SageMaker.

First, a training script (train.py) was created containing the model architecture, training loop, and model saving logic. The model was saved using the SageMaker environment variable SM_MODEL_DIR, which ensures that the trained weights are properly stored and packaged as model.tar.gz in Amazon S3.

A PyTorch Estimator was then configured specifying:

- Entry point script

- Instance type (ml.m5.large)

- Framework version

IAM execution role

The training job was launched using estimator.fit(), and once completed successfully, the trained model artifact was stored automatically in an S3 bucket.

After training, the model was deployed to a real-time inference endpoint using estimator.deploy(). This created:

- A SageMaker model

- An endpoint configuration

- A live endpoint in "InService" state

Finally, inference was tested by sending a sample input tensor with shape (1, 3, 32, 32), corresponding to CIFAR-10 image dimensions. The endpoint returned a predicted class index, confirming successful deploymen

The endpoint was deleted after testing to prevent unnecessary resource usage and additional AWS charges.