import numpy as np
import torch
import gym
import argparse
import os
import dm_control
import dmc2gym
import utils
import TD3
import OurDDPG
import DDPG
from einops import rearrange, reduce, repeat
count = 0
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--domain_name", default="hopper")
parser.add_argument("--task_name", default="stand")
parser.add_argument("--seed", default=233, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("--policy_utd", default=1, type=int)  # Frequency of delayed policy updates
parser.add_argument("--policy_interval", default=1, type=int)  # Frequency of delayed policy updates
parser.add_argument("--reset_freq", default=20000, type=int)  # Frequency of delayed policy updates
parser.add_argument("--qnum", default=5, type=int)  # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--N",default=1,type = int)
parser.add_argument("--bili",default=2,type = int)
args = parser.parse_args()
def create_directory_if_not_exists(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  
        print(f"Created directory: {path}")  
    else:  
        print(f"Directory already exists: {path}")
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, domain_name, task_name, seed, eval_episodes=10):
    eval_env = make_env(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed)
    eval_env.seed(seed + 100)
    global count
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action1(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    with open(f"seed={args.seed}_result.txt", "a") as file:
        file.write(f"Timesteps {count} Evaluation over {eval_episodes} episodes: {avg_reward:.3f}\n")
    count += 5000
    return avg_reward

def make_env(domain_name, task_name, seed):
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed, from_pixels=False,  visualize_reward=False)
    return env

if __name__ == "__main__":
    
    with open(f"seed={args.seed}_result.txt", "a") as file:
        file.write(f"policy:{args.policy} Domain: {args.domain_name} Task: {args.task_name} seed:{args.seed} policy_utd:{args.policy_utd} policy_interval:{args.policy_interval} reset_freq:{args.reset_freq} qnum:{args.qnum} N:{args.N} bili:{args.bili}\n")
    file_name = f"{args.policy}_Domain: {args.domain_name}_Task: {args.task_name}_{args.seed}_{args.policy_utd}_{args.policy_interval}_{args.reset_freq}_{args.qnum}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Domain: {args.domain_name}, Task: {args.task_name}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = make_env(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "Q_num": args.qnum,
        "reset_freq": args.reset_freq * args.policy_utd,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    utd = args.policy_utd
    interval = args.policy_interval

    # Evaluate untrained policy
    evaluations = [eval_policy(policy,args.domain_name,args.task_name, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    gradient_steps = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()  # 随机选择一个动作
        else:
            if (t % args.bili == 0):
                action = (policy.select_action1(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
            else:
                actions = None
                states = None
                for i in range(args.N):
                    action = (policy.select_action2(np.array(state))
                            + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)
                    if i == 0:
                        actions = torch.from_numpy(action).unsqueeze(0)
                        states = torch.from_numpy(state).unsqueeze(0)
                    else:
                        actions = torch.cat([actions,torch.from_numpy(action).unsqueeze(0)],dim=0)
                        states = torch.cat([states,torch.from_numpy(state).unsqueeze(0)],dim=0)
                #print(states.float(),actions.float())
                T_Q = policy.critic(states.float().to('cuda'),actions.float().to('cuda'))
                T_Q = rearrange(T_Q,'i j k -> (j k) i')
                Qmax_std = 0
                Q_index = -1
                Qmax_index = 0
                T_Q = T_Q.tolist()
                for i in T_Q:
                    Q_std = np.std(i)
                    Q_index += 1
                    if Q_std > Qmax_std:
                        Qmax_std = Q_std
                        Qmax_index = Q_index
                action = actions[Qmax_index]
                action = np.array(action)
                #raise ValueError("EVAstop")
                
                
        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward
        results = None
        # Train agent after collecting sufficient data
        if (t >= args.start_timesteps and t % interval == 0):
            for i in range(interval * utd):
                results = policy.train(replay_buffer, args.batch_size)
                gradient_steps += 1

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Gradient Time: {gradient_steps} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.5f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if results is not None:
                print(results)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.domain_name,args.task_name, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
