import tqdm
from EnvBatch import EnvBatch
from hyperparameters import *
from eval import *
from Agent import Agent
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

agent = Agent(number_actions)

env_batch = EnvBatch(number_environments)
batch_states = env_batch.reset()

with tqdm.trange(0, 3001) as progress_bar:
    for i in progress_bar:
        batch_actions = agent.act(batch_states)
        batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
        batch_rewards *= 0.01
        agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
        batch_states = batch_next_states
        if i % 1000 == 0:
            print("Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))


# def show_video_of_model(agent, env):
#   state, _ = env.reset()
#   done = False
#   frames = []
#   while not done:
#     frame = env.render()
#     frames.append(frame)
#     action = agent.act(state)
#     state, reward, done, _, _ = env.step(action[0])
#   env.close()
#   imageio.mimsave('video.mp4', frames, fps=30)

# show_video_of_model(agent, env)

# def show_video():
#     mp4list = glob.glob('*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")

# show_video()