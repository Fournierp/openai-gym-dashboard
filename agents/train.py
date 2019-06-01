import json
from sklearn.model_selection import ParameterGrid
from agents.cartpole import CartpoleAgent

if __name__ == '__main__':
    # Load JSON
    with open('hyperparameters.json') as f:
        data = json.load(f)
        # Make a Grid Search
        grid = ParameterGrid(data)
        for i, params in enumerate(grid):
            print(params)
            agent = CartpoleAgent(episodes=params["episodes"], activation=params['activation'],
                                  batch_size=params['batch_size'], epsilon_decay=params['epsilon_decay'],
                                  gamma=params['gamma'], lr=params["lr"], neurons=params["neurons"])
            agent.play()
            agent.save_model()
