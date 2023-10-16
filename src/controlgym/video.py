import os
import glob
import io
import base64
from pathlib import Path

from IPython.display import HTML
from IPython import display as ipythondisplay

from gym import Env
from gym.logger import set_level
from gym.wrappers.record_video import RecordVideo

import numpy as np

from controlgym.agents.agent import Agent


def show_video(max_number=-1, videos_folder="videos"):
    mp4list = glob.glob(f"{videos_folder}/*.mp4")
    if len(mp4list) > 0:
        n_videos = len(mp4list)
        if max_number > -1:
            n_videos = min(max_number, n_videos)

        for i in range(n_videos):
            mp4 = mp4list[i]
            video = io.open(mp4, "r+b").read()
            encoded = base64.b64encode(video)
            ipythondisplay.display(
                HTML(
                    data="""<video alt="test" autoplay
                        loop controls style="height: 400px;">
                        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>""".format(
                        encoded.decode("ascii")
                    )
                )
            )
    else:
        print("Could not find video")


def record_video(env: Env, agent: Agent, folder: Path = Path("./videos")):
    os.makedirs(folder, exist_ok=True)
    set_level(40)
    observations = []
    actions = []

    has_time = hasattr(env, "dt")
    times = []

    monitored_env = RecordVideo(
        env, folder.absolute(), episode_trigger=lambda episode_number: True
    )
    observation = monitored_env.reset()
    agent.reset()
    done = False
    time = 0
    while not done:
        monitored_env.render()
        action = agent.act(observation)
        observation, _reward, done, _info = monitored_env.step(action)
        observations.append(observation)
        actions.append(action)

        if has_time:
            time += env.dt
            times.append(time)

    return np.array(observations), np.array(actions), np.array(times)
