from agents.cartpole import CartpoleAgent

if __name__ == '__main__':
    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.99, neurons=[8, 8, 8])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.99, neurons=[16, 8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.99, neurons=[8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.99, neurons=[16, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.99, neurons=[8])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.99, neurons=[8, 8, 8])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.97, neurons=[16, 8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.97, neurons=[8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.999, gamma=0.97, neurons=[16, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.99, neurons=[8, 8, 8])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.99, neurons=[16, 8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.99, neurons=[8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.99, neurons=[16, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.99, neurons=[8])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.99, neurons=[8, 8, 8])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.97, neurons=[16, 8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.97, neurons=[8, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.97, neurons=[16, 4])
    agent.play()
    agent.save_model()

    agent = CartpoleAgent(episodes=10000, epsilon_decay=0.995, gamma=0.97, neurons=[8])
    agent.play()
    agent.save_model()
