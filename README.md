# openai-gym-dashboard

## Introduction

OpenAI Gym Dashboard is a web app designed with Plotly's Dashboard Tool. This lightweight interface is a great tool to visualize the performance of your Reinforcement Learning or Artificial Intelligent Agent while it is training or compare multiple trained Agents. 

## Installation

Firstly, ensure that you have pip install. In which case follow these steps using the command line:

```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

Then install the required libraries listed in the requirements.txt
```
pip install -r requirements.txt
```

## Usage Example
To launch the Dash web app, run the following command-line:
```
python index.py
```

To start the training of the agent:
```
python agents/cartpole.py
```

To start the training of multiple agents with grid search, change the **hyperparameters.json** file and run:
```
python agents/train.py
```

## Release history

* 1.0
    * Live model visualizations

## Built With

* [Dash](https://github.com/plotly/dash/) Analytical Web Apps for Python
* [openai-gym](https://github.com/openai/gym) A toolkit for developing and comparing reinforcement learning algorithms


## Authors

* **Paul Fournier** - *Initial work* - [Fournierp](https://github.com/Fournierp)


## License

This project is licensed under the Apache License - see the LICENSE.md file for details
