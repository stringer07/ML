import gym

env = gym.make('CartPole-v1',render_mode="rgb_array")
state = env.reset()

for t in range(100):
    env.render()
    
    action = env.action_space.sample()
    state, reward, done, info,kk = env.step(action)
    
    print(state)  # Moved this line here to print the updated state
    
    if done:
        print('Finished')
        break

env.close()
