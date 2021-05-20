# Add the training and testing folders here in the same folder.
files:
trainer.py
config_0.py
Folder name: 'Train' (Training data folder)
Folder name: 'Test' (Testing data folder)

# In the command line :
Dump default configurations to the config.yaml file as reference
'''
python trainer.py --print_config>config.yaml
'''

# To change the default configurations, open the config.yaml file and change the default data
'''
nano config.yaml
'''

# Default model parameters that are configurable:
'''
model.learning_rate: 0.001 (Can be configured in the code)
model.training_dataset_folder: train (training data folder)
model.testing_dataset_folder: test (testing data folder)
model.batch_size: 4
'''

# To run the model with those changed hyperparameters, type : (in the command line)
'''
python trainer.py --config config.yaml --trainer.max_epochs 200
'''

# To load the tensorboard
'''
tensorboard --logdir ./lightning_logs
'''

open the link that is given in the command line. Most probably it will be http://localhost:6006/.

We can visualise the training and testing loss and accuracy per step and per epoch.
