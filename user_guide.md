# DeepM

## Config

Settings for all modules are stored in the config file. 

## Command Line Interface

- Downloading data

To download data, run `python main.py --mode=load_data`. This will save the global data matrix as an `.npy` file and will be loaded during training and test time. 

- Training an agent

To train an agent, run `python main.py --mode=train`. This will train an agent from scratch. To edit training settings, update config parameters under agent. The policy weights will be saved in the file `agent.pt`. 

- Running a backtest

To test the model, run `python main.py --mode=backtest`. This will give a performance summary over training and test data. 