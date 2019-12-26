import tensorflow as tf
import numpy as np
import os
import sys
import shutil
from collections import OrderedDict
import pdb
from matplotlib import pyplot as plt

from config import config
from policy import Policy
from value import Value
from discriminator import Discriminator
from posterior import Posterior
from buffer import Buffer
import utils as U
from pdb import set_trace as db

dir_name = os.path.dirname(__file__)

sys.path.append(os.path.join(dir_name, '../env/'))
import circle_theta as env

np.set_printoptions(precision = 4)


def load_expert_traj():
    loaded_dict = np.load(config.expert_traj_path, allow_pickle = True).item()
    traj_np = loaded_dict['traj']
    theta_np = loaded_dict['theta']
    print()
    print('expert trajectories shape: ', traj_np.shape)
    print('expert thetas shape: ', theta_np.shape)
    print()
    
    return traj_np, theta_np


def stat_collect(*args):
    for i in range(len(args) // 2):
        args[2 * i].append(args[2 * i + 1])


def print_and_clear(*ds):
    for d in ds:
        for k, v in d.items():
            print('{}: {:.4f}'.format(k, np.mean(v)))
            v.clear()
        print()


def set_target_net_update():
    policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Policy_stu/')
    old_policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Policy_old/')
    old_policy.set_update_op(policy_vars, old_policy_vars)
    old_policy.run_update_op()

    value_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Value_stu/')
    old_value_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Value_old/')
    old_value_net.set_update_op(value_net_vars, old_value_net_vars)
    old_value_net.run_update_op()


def bc_pretrain():
    losses = []
    for i in range(1, config.itr + 1):
        expert_traj_state, expert_traj_action = buffer.sample_expert_traj()
        expert_traj_code = buffer.sample_code()
        loss = policy.bc_train(expert_traj_state, expert_traj_code, expert_traj_action)
        losses.append(loss)

        if i % config.print_itr == 0:
            print('[t{}] bc pretrain loss: {}'.format(i, np.mean(losses)))
            print()
            losses.clear()

            _, stu_traj_action, stu_traj_code = buffer.sample_stu_traj()
            file_name = os.path.join(config.draw_traj_dir, ckpt_name, '{}_bc_{}.{}'.format(ckpt_name, i, config.draw_extension))
            env.draw_traj_theta(stu_traj_action, stu_traj_code, file_name)
        
        if i % config.save_itr == 0:
            saver.save(sess, config.save_path)
            print('ckpt saved \n')


def sample_draw():
    _, stu_traj_action, stu_traj_code = buffer.sample_stu_traj()
    to_save = {'action': stu_traj_action, 'code': stu_traj_code}
    np.save(os.path.join(config.draw_traj_dir, ckpt_name, '{}_sample'.format(ckpt_name)), to_save)
    file_name = os.path.join(config.draw_traj_dir, ckpt_name, '{}_sample.{}'.format(ckpt_name, config.draw_extension))
    env.draw_traj_theta(stu_traj_action, stu_traj_code, file_name)


def gae(stu_traj_state, stu_traj_action, stu_traj_code):
    dis_score = discriminator.get_stu_score(stu_traj_state, stu_traj_action)
    log_p = posterior.get_log_posterior(stu_traj_state, stu_traj_action, stu_traj_code)
    r = config.dis_coef * dis_score + config.post_coef * log_p

    value = value_net.get_value(stu_traj_state, stu_traj_code)

    value_spvs_reversed = []
    adv_reversed = []

    next_value = None
    for t in reversed(range(config.max_traj_len)):
        curr_r = r[:, t]
        curr_value = value[:, t]
        
        if t == config.max_traj_len - 1:
            value_spvs = curr_r
            adv = curr_r - curr_value
        else:
            td = curr_r + config.gamma * next_value - curr_value
            value_spvs = curr_r + config.gamma * (config.lam * value_spvs_reversed[-1] + (1 - config.lam) * next_value)
            adv = td + config.gamma * config.lam * adv_reversed[-1]

        value_spvs_reversed.append(value_spvs)
        adv_reversed.append(adv)

        next_value = curr_value

    value_spvs = np.array(list(reversed(value_spvs_reversed))).T
    adv = np.array(list(reversed(adv_reversed))).T

    dis_score_mean = np.mean(dis_score)
    log_p_mean = np.mean(log_p)

    return value_spvs, adv, dis_score_mean, log_p_mean


def main():
    traj_np, theta_np = load_expert_traj()

    global sess, policy, old_policy, value_net, old_value_net, \
        posterior, discriminator, buffer, saver, ckpt_name
    
    sess = U.get_tf_session()
    policy = Policy(config, sess, 'stu')
    old_policy = Policy(config, sess, 'old')
    value_net = Value(config, sess, 'stu')
    old_value_net = Value(config, sess, 'old')
    posterior = Posterior(config, sess)
    discriminator = Discriminator(config, sess)
    buffer = Buffer(config, traj_np, theta_np, policy)
    saver = tf.train.Saver()
    ckpt_name = os.path.splitext(os.path.basename(config.save_path))[0]

    if os.path.exists(config.load_path + '.index'):
        saver.restore(sess, config.load_path)
        print('\nloaded from load_path \n')
    else:
        print('\nload_path does not exist \n')
        init = tf.global_variables_initializer()
        sess.run(init)

    if config.mode == 'draw':
        sample_draw()
        exit('finish sample drawing')

    elif config.mode == 'bc':
        bc_pretrain()
        exit('finish bc pretraining')

    set_target_net_update()

    dis_losses = []
    dis_expert_scores = []
    dis_stu_scores = []
    post_losses = []
    post_values = []
    dis_scores = []
    log_ps = []
    stu_values = []
    stu_advs = []
    policy_losses = []
    policy_rewards = []
    entropies = []
    action_log_prob_olds = []
    policy_clipped_freqs = []
    value_losses = []
    old_values = []
    values = []
    value_clipped_freqs = []

    # ratios = []

    dis_stat = OrderedDict([
        ('dis loss', dis_losses),
        ('dis export score', dis_expert_scores),
        ('dis student score', dis_stu_scores)
    ])

    post_stat = OrderedDict([
        ('posterior loss', post_losses),
        ('posterior', post_values)
    ])

    gae_stat = OrderedDict([
        ('dis score in GAE', dis_scores),
        ('log p in GAE', log_ps),
        ('value from GAE', stu_values),
        ('adv from GAE', stu_advs),
    ])

    policy_stat = OrderedDict([
        ('policy loss', policy_losses),
        ('policy reward', policy_rewards),
        ('entropy', entropies),
        ('policy clipped freq', policy_clipped_freqs),
        ('action log p old', action_log_prob_olds),
    ])

    value_stat = OrderedDict([
        ('value loss', value_losses),
        ('value', values),
        ('value clipped freq', value_clipped_freqs),
        ('value old', old_values)
    ])

    for i in range(1, config.itr + 1):

        for j in range(config.inner_itr_1):
            stu_traj_state, stu_traj_action, stu_traj_code = buffer.sample_stu_traj()
            expert_traj_state, expert_traj_action = buffer.sample_expert_traj()
            
            dis_loss, dis_expert_score, dis_stu_score = discriminator.train(expert_traj_state, expert_traj_action, stu_traj_state, 
                stu_traj_action)
            post_loss, post_value = posterior.train(stu_traj_state, stu_traj_action, stu_traj_code)

            if j == config.inner_itr_1 - 1:
                stat_collect(dis_losses, dis_loss, dis_expert_scores, dis_expert_score, dis_stu_scores, dis_stu_score)
                stat_collect(post_losses, post_loss, post_values, post_value)
        
        for k in range(config.inner_itr_2):
            stu_traj_state, stu_traj_action, stu_traj_code = buffer.sample_stu_traj()

            stu_traj_value_spvs, stu_traj_adv, dis_score, log_p = gae(stu_traj_state, stu_traj_action, stu_traj_code)
            action_log_prob_old = old_policy.get_action_log_prob(stu_traj_state, stu_traj_code, stu_traj_action)
            stu_traj_value_old = old_value_net.get_value(stu_traj_state, stu_traj_code)

            policy_loss, policy_reward, entropy, policy_clipped_freq = policy.train(stu_traj_state, stu_traj_code, stu_traj_action, 
                stu_traj_adv, action_log_prob_old)

            # ratios.extend(ratio)
            value_loss, value, value_clipped_freq = value_net.train(stu_traj_state, stu_traj_code, stu_traj_value_spvs, stu_traj_value_old)
            
            if k == config.inner_itr_2 - 1:
                stat_collect(
                    dis_scores, dis_score, 
                    log_ps, log_p, 
                    stu_values, stu_traj_value_spvs, 
                    stu_advs, stu_traj_adv, 
                    policy_losses, policy_loss,
                    policy_rewards, policy_reward, 
                    entropies, entropy,
                    policy_clipped_freqs, policy_clipped_freq, 
                    action_log_prob_olds, action_log_prob_old, 
                    value_losses, value_loss,
                    values, value,
                    value_clipped_freqs, value_clipped_freq,
                    old_values, stu_traj_value_old
                )

        if i % config.print_itr == 0:
            print('[t{}]'.format(i))
            print_and_clear(dis_stat, post_stat, gae_stat, policy_stat, value_stat)

            file_name = os.path.join(config.draw_traj_dir, ckpt_name, '{}_{}.{}'.format(ckpt_name, i, config.draw_extension))
            env.draw_traj_theta(stu_traj_action, stu_traj_code, file_name)

        if i % config.update_period == 0:
            old_policy.run_update_op()
            old_value_net.run_update_op()
        
        if i % config.save_itr == 0:
            saver.save(sess, config.save_path)
            print('ckpt saved\n')


if __name__ == "__main__":
    main()
