from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer

from BSE2agent import Agent2BSE

class BSEEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None,**kwargs):
        self.env = Agent2BSE(**kwargs)

        self.render_mode = render_mode

        self.min_nor = 0
        self.max_nor = 1

        self.minprice=self.env.minprice
        self.maxprice=self.env.maxprice

        # self.lob = np.array(
        #     [np.ones((self.windows,self.env.len_lobs)),
        #     np.zeros((self.windows,self.env.len_lobs))]
        #     )
        
        obs, rew, done,info = self.env.reset()

        # high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api

        self.action_space = spaces.Box(
            low=self.min_nor, high=self.max_nor, shape=(1,), dtype=np.float32
        )

        obs = self.obstrans(obs)
        high = np.ones_like(obs)
        low = np.zeros_like(obs)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def obstrans(self,obs_ori):
        
        # lob_ask, lob_bid = obs_ori[0], obs_ori[1]
        
        # obs = lob_ask+lob_bid
        # obs=np.array(obs,dtype=np.float32)
        # obs = (obs-self.minprice)/(self.maxprice-self.minprice)

        
        
        # lob_ask=np.array(lob_ask,dtype=np.float32)
        # lob_ask = (lob_ask-self.minprice)/(self.maxprice-self.minprice)
        # lob_bid=np.array(lob_bid,dtype=np.float32)
        # lob_bid = (lob_bid-self.minprice)/(self.maxprice-self.minprice)
        # lob_append = np.array([[lob_ask],[lob_bid]])

        obs=np.array(obs_ori,dtype=np.float32)
        obs = (obs-self.minprice)/(self.maxprice-self.minprice)


        return obs

    def step(self, action):
        # action = action*(self.maxprice-self.minprice)+self.minprice
        obs, rew, done,info = self.env.step(action)

        obs = self.obstrans(obs)
        # print(obs)

        return  obs, rew, done, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        obs, rew, done, info = self.env.reset()

        obs = self.obstrans(obs)

        # return  obs, rew, done,info
        
        if not return_info:
            return obs
        else:
            return obs, {}


    def render(self, mode="human"):
        pass
        # if self.render_mode is not None:
        #     return self.renderer.get_renders()
        # else:
        #     return self._render(mode)
