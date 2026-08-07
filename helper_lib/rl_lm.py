import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import words


def build_vocab(vocab_size=1000, min_len=3, max_len=8, seed=42):
    """Samples a fixed vocabulary from nltk's word list, matching Module 10's
    build_a_vocab. Deterministic given the same seed, so the API can rebuild
    the exact same word_to_idx/idx_to_word mapping used at training time."""
    all_words = words.words()
    filtered = [w.lower() for w in all_words if w.isalpha() and min_len <= len(w) <= max_len]
    unique = sorted(set(filtered))
    random.seed(seed)
    vocab = random.sample(unique, vocab_size)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word


class SimpleNet(nn.Module):
    """The base next-word model from Module 10, unchanged."""

    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = self.activation(x)
        return self.fc2(x)


def is_valid_transition(prev, curr):
    return curr[0] == prev[-1]


def encode_sequence_state(context, vocab_size, context_size=3):
    vec = torch.zeros(context_size * vocab_size, dtype=torch.float32)
    last_k = context[-context_size:] if len(context) >= context_size else context
    offset = context_size - len(last_k)
    for i, idx in enumerate(last_k):
        vec[(offset + i) * vocab_size + idx] = 1.0
    return vec


def generate_text_from_base_model(context, model, vocab, idx_to_word):
    """Encodes the context, gets next-word logits from model, and samples among
    only the words that satisfy the word-chain constraint."""
    encoded_context = encode_sequence_state(context, len(vocab)).unsqueeze(0)
    next_word_logits = model(encoded_context)

    admissible_idxs = [i for i, w in enumerate(vocab) if is_valid_transition(idx_to_word[context[-1]], w)]
    next_word_probabilities = F.softmax(next_word_logits[0][admissible_idxs], dim=-1)
    next_word_dist = torch.distributions.Categorical(next_word_probabilities)
    next_word_sample = next_word_dist.sample()
    return admissible_idxs[next_word_sample.item()], next_word_dist.log_prob(next_word_sample)


def shaped_reward(sequence, end_letter="e"):
    """Rewards word-chain validity at each step, plus a large bonus/penalty for
    whether the final word ends in end_letter. This end-letter bonus is the
    "specific format" signal Assignment 5 post-trains for."""
    reward = 0.0
    for i in range(1, len(sequence)):
        if is_valid_transition(sequence[i - 1], sequence[i]):
            reward += 1.0
        else:
            reward -= 3.0

    if sequence[-1][-1] == end_letter:
        reward += 50
    else:
        reward -= 10

    return reward


class Environment:
    """Same state machine as Module 10's Environment, with two changes:

    1. The start word is always a fixed word (the "opening" of the format)
       instead of a random one, since it is a precondition of the format
       rather than something the agent chooses or needs to learn.
    2. Episodes always run for exactly max_seq_length words instead of ending
       the instant *any* word in the chain happens to end in end_letter.
       Module 10's original condition (stop as soon as the target letter is
       hit, wherever in the chain) makes the trained policy collapse onto
       "fixed_start -> single lucky word", since the reward is far more
       reliable if it's grabbed at the first opportunity -- e.g. with
       start_word="group" it converges to always answering "group picidae"
       and nothing else. Only checking the *last* word of a full-length chain
       is what "answer in a specific format" actually calls for -- a
       complete, fixed-length response that happens to end correctly,
       not a chain that stops the moment it can.
    """

    def __init__(self, word_to_idx, idx_to_word, max_seq_length, end_letter="e", context_size=3):
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.max_seq_length = max_seq_length
        self.end_letter = end_letter
        self.context_size = context_size
        self.state = []

    def reset(self, start_word):
        self.state = [self.word_to_idx[start_word]]
        return self.state[-self.context_size:]

    def step(self, action):
        self.state = self.state + [action]
        new_context = self.state[-self.context_size:]
        reward, done = 0, False
        if len(self.state) == self.max_seq_length:
            done = True
            sequence = [self.idx_to_word[i] for i in self.state]
            reward = shaped_reward(sequence, self.end_letter)
        return new_context, reward, done


class Policy:
    """REINFORCE policy-gradient training, adapted from Module 10's Policy class.

    Module 10's own train_one_epoch collects a log-prob per action into
    batch_logp but then never uses it -- it calls compute_loss with the bare
    loop variable `logp`, which by that point only holds the *last* action's
    log-prob, so gradients only ever flow through one action per epoch. This
    version stacks the full batch_logp so every collected action contributes
    to the policy gradient, matching the REINFORCE algorithm the notebook
    itself cites (https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).
    """

    def __init__(self, model, optimizer, env, vocab, idx_to_word, start_word, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.vocab = vocab
        self.idx_to_word = idx_to_word
        self.start_word = start_word
        self.batch_size = batch_size

    def get_action(self, obs):
        return generate_text_from_base_model(obs, self.model, self.vocab, self.idx_to_word)

    def compute_loss(self, logp, weights):
        return -(logp * weights).mean()

    def train_one_epoch(self):
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []
        batch_logp = []

        obs = self.env.reset(self.start_word)
        ep_rews = []

        while True:
            act, logp = self.get_action(obs)
            obs, rew, done = self.env.step(act)

            batch_acts.append(act)
            ep_rews.append(rew)
            batch_logp.append(logp)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_weights += [ep_ret] * ep_len

                obs, ep_rews = self.env.reset(self.start_word), []

                if len(batch_acts) > self.batch_size:
                    break

        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(
            torch.stack(batch_logp),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss, batch_rets, batch_lens

    def generate_chain(self):
        """Samples one full word chain from the current policy, starting from
        start_word, for demo/inference use (e.g. the API endpoint)."""
        context = self.env.reset(self.start_word)
        chain = [self.start_word]
        done = False
        while not done:
            action, _ = self.get_action(context)
            context, _, done = self.env.step(action)
            chain.append(self.idx_to_word[context[-1]])
        return chain
