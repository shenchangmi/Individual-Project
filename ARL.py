import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.trainer.utils import gather_info, test_episode
from tianshou.utils.net.continuous import ActorProb, Critic


from torch import nn
from tianshou.utils.net.common import MLP


from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

ModuleType = Type[nn.Module]

# args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device
class conv(nn.Module):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        hidden_sizes_conv: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1

        self.model_conv = nn.Sequential(
            nn.Conv2d(state_shape[0],hidden_sizes_conv[0],5),
            nn.ReLU(),
            nn.Conv2d(hidden_sizes_conv[0],hidden_sizes_conv[1],5),
            nn.ReLU()
        )
        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        obs2mlp = self.model_conv(obs)

        return obs2mlp

class Net_conv(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        hidden_sizes_conv: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1

        self.model_conv = nn.Sequential(
            nn.Conv2d(state_shape[0],hidden_sizes_conv[0],3),
            nn.ReLU(),
            nn.Conv2d(hidden_sizes_conv[0],hidden_sizes_conv[1],3),
            nn.ReLU()
        )

        
        # self.model_rnn = nn.LSTM(state_shape[1], state_shape[1]*2)
        
        obs = np.array([np.zeros(state_shape)])
        if self.device is not None:
            obs = torch.as_tensor(obs, dtype=torch.float32)

        output_dim_conv = self.model_conv(obs).shape
        # output_dim_conv=(hidden_sizes_conv[1],11,11)
        input_dim = int(np.prod(output_dim_conv))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0

        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        obs2mlp = self.model_conv(obs)
        obs2mlp.flatten()
        logits = self.model(obs2mlp)
        bsz = logits.shape[0]

        if self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms) # reshape
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

class Net_rnn(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        hidden_sizes_conv: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1

        self.rnn_hidden_size = state_shape[1]
        self.batch_size = 1

        self.model_rnn = nn.LSTM(state_shape[1], self.rnn_hidden_size)
        
        obs = np.array([np.zeros(state_shape)])
        if self.device is not None:
            obs = torch.as_tensor(obs, dtype=torch.float32)

        input_dim = int(np.prod(self.rnn_hidden_size))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0

        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.output_dim = self.model.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            
        batch_size = obs.shape[0]
        length = obs.shape[1]
        input_size = obs.shape[2]
        # h0 = torch.zeros(1, self.batch_size,  self.rnn_hidden_size).to(self.device) 
        # c0 = torch.zeros(1, self.batch_size,  self.rnn_hidden_size).to(self.device)
        h0 = torch.zeros(1, batch_size,  self.rnn_hidden_size).to(self.device) 
        c0 = torch.zeros(1, batch_size,  self.rnn_hidden_size).to(self.device)

        obs_t = torch.zeros(length,batch_size,input_size).to(self.device)

        for i in range(batch_size):
            obs_t[:,i,:] = obs[i,:,:]
        
        # Forward propagate LSTM
        # out, _ = self.lstm(obs, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        output, (h_n, c_n) = self.model_rnn(obs_t, (h0, c0))
        obs2mlp = h_n[0]
        obs2mlp.flatten()
        logits = self.model(obs2mlp)
        bsz = logits.shape[0]

        if self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms) # reshape
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='BSE-v1')
    # parser.add_argument('--task', type=str, default='Pendulum-v1')
    # Pendulum-v1
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=600000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95) #0.95
    parser.add_argument('--epoch', type=int, default=200)
    # parser.add_argument('--step-per-epoch', type=int, default=3000*32)
    parser.add_argument('--step-per-epoch', type=int, default=100*8*128) # traing中一个epoch需要env step数据
    parser.add_argument('--episode-per-collect', type=int, default=8) # train_collect一次collect采集几个episode
    parser.add_argument('--repeat-per-collect', type=int, default=2) # 学习时重复几次
    parser.add_argument('--batch-size', type=int, default=128) # 一次梯度更次需要的batch_size
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[1024,256,64,16]) # [32,16]
    # parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128,32,16]) # [32,16]
    parser.add_argument('--hidden-sizes-conv', type=int, nargs='*', default=[2,4]) #[20,64]
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=1) #5
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('---resume', type=bool, default=False)
    # parser.add_argument('---resume', type=bool, default=True)
    parser.add_argument('--watch', type=bool, default=False)
    # parser.add_argument('--watch', type=bool, default=True)
    parser.add_argument("--save-interval", type=int, default=4)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    if args.reward_threshold is None:
        default_reward_threshold = {"Pendulum-v0": -250, "Pendulum-v1": -250}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task,training=False) for _ in range(args.test_num)]
    )

    # test_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task,training=False, range2_input = 150) ,
    #     lambda: gym.make(args.task,training=False, range2_input = 200) ,
    #     lambda: gym.make(args.task,training=False, range2_input = 250) ,
    #     lambda: gym.make(args.task,training=False, range2_input = 300) ,
    #     lambda: gym.make(args.task,training=False, range2_input = 350)]
    # )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # TODO net model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    # net = Net_conv(args.state_shape, hidden_sizes=args.hidden_sizes, hidden_sizes_conv=args.hidden_sizes_conv, device=args.device)
    net = Net_rnn(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device) # net输出后logits经MLP生成mu，sigma
    critic = Critic(
        Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
        device=args.device
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        # return Independent(Normal(*logits), 1)
        return Independent(Normal(logits[0],torch.as_tensor([[0.1]],device=args.device)), 1)
        

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space,
    )
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)

    test_collector = [gym.make(args.task,training=False,range2_input = price) for price in [150,200,250,300,350]]
    # test_collector = [gym.make(args.task,training=False,range2_input = price) for price in [350]]
    # test_collector = [gym.make(args.task,training=False,traders_per = percent) 
    #     for percent in [[18,10,2,18,10,2],[2,10,18,2,10,18],
    #                     [10,2,18,10,2,18],[10,18,2,10,18,2],
    #                     [18,2,10,18,2,10],[2,18,10,2,18,10]]]




    # watch = True
    # args.watch = True
    # args.resume = True

    if not args.watch:

        from datetime import datetime
        current_time = datetime.now().strftime("%m%d_%H-%M-%S")
        # log
        log_file = f'BSE_ppo_{current_time}'
        log_path = os.path.join(args.logdir, args.task, "ppo", log_file)
        writer = SummaryWriter(log_path, log_file)
        writer.add_text("args", str(args))
        logger = TensorboardLogger(writer, save_interval=4)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, f"policy.pth"))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(log_path, f"checkpoint.pth")
            # Example: saving by epoch num
            # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                    "ret_rms": [policy.ret_rms.count,
                        policy.ret_rms.mean,
                        policy.ret_rms.var,],
                }, ckpt_path
            )
            return ckpt_path

        if args.resume:
            # load from existing checkpoint
            print(f"Loading agent under {log_path}")
            # ckpt_path = os.path.join(log_path, "checkpointSep28_08-33-21.pth")
            ckpt_path = 'log\BSE-v1\ppo\BSE_ppo_1002_10-24-24\checkpoint.pth'
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=args.device)
                policy.load_state_dict(checkpoint["model"])
                optim.load_state_dict(checkpoint["optim"])
                print("Successfully restore policy and optim.")
            else:
                print("Fail to restore policy and optim.")


        # trainer
        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,  # 即episode_per_test
            args.batch_size,
            episode_per_collect=args.episode_per_collect, #train
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=args.resume,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        for epoch, epoch_stat, info in trainer:
            print(f"Epoch: {epoch}")
            print(epoch_stat)
            print(info)

        # logger.close()
        # assert stop_fn(info["best_reward"])

        if __name__ == "__main__":
            pprint.pprint(info)
            # Let's watch its performance!
            env = gym.make(args.task)
            policy.eval()
            collector = Collector(policy, env)
            result = collector.collect(n_episode=1, render=args.render)
            # result = collector.collect(n_episode=1, render=0)
            rews, lens = result["rews"], result["lens"]
            print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

    else:

        best_policy =  torch.load('log\BSE-v1\ppo\BSE_ppo_1023_19-31-02\checkpoint.pth', map_location=args.device)
        policy.load_state_dict(best_policy["model"])

        best_actor =  torch.load('log\BSE-v1\ppo\BSE_ppo_1016_17-46-03\policy.pth', map_location=args.device)
        policy.load_state_dict(best_actor)

        

        # test_envs = DummyVectorEnv(
        #     [lambda: gym.make(args.task,training=False) for _ in range(1)]
        # )
        # test_envs.seed(args.seed)
        # test_collector = Collector(policy, test_envs)
        # test_collector.reset_stat()
        # test_result = test_episode(
        #     policy, test_collector, None, 0,
        #     args.test_num, None, 0, None
        # )

        #TODO 固定范围测试
        # test_envs = [gym.make(args.task,training=False,range2_input = price) for price in [150,200,250,300,350]]
        # test_envs = [gym.make(args.task,training=False,range2_input = price) for price in [150,]]
        # test_envs.seed(args.seed)
        # test_collector = Collector(policy, test_envs)
        # test_collector.reset_stat()
        # test_result = test_episode(
        #     policy, test_envs, None, 0,
        #     5, None, 0, None
        # )
        # test_result = test_episode(
        #     policy, test_collector, None, 0,
        #     5, None, 0, None
        # )
        # test_collector = [gym.make(args.task,training=False,range2_input = price) for price in [150 for _ in range(5)]]
        # test_collector = [gym.make(args.task,training=False,traders_per = percent) 
        #     for percent in [[18,10,2,18,10,2],[2,10,18,2,10,18],
        #                     [10,2,18,10,2,18],[10,18,2,10,18,2],
        #                     [18,2,10,18,2,10],[2,18,10,2,18,10]]]

        # test_collector = [gym.make(args.task,training=False,traders_per = [18,2,10,18,2,10]) 
        #     for _ in range(5)]

        test_result = test_episode(
            policy, test_collector, None, 0,
            1, None, 0, None
        )

        print(test_result)

def test_ppo_resume(args=get_args()):
    args.resume = True
    test_ppo(args)


if __name__ == "__main__":
    test_ppo()