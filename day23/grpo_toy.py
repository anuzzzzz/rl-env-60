import numpy as np
import matplotlib.pyplot as plt

REWARDS = np.array([10.0, 2.0])   # answer A = index 0, answer B = index 1
rng = np.random.default_rng(0)    # fixed seed, so runs are reproducible


def softmax(z):
    # TODO: turn logits into probabilities. Subtract max(z) first for stability.
    ...
    z_shifted = z - np.max(z)

    z_raised = np.exp(z_shifted)
    total = np.sum(z_raised)
    return z_raised/total


def sample_group(pi, G):
    # pi: the policy's probabilities, e.g. [0.3, 0.7].
    # G:  how many answers to sample for this group, e.g. 8.

    # rng.choice(n, size, p) picks `size` numbers from 0..n-1,
    # where number k is picked with probability p[k].
    # Here, n = len(pi) = 2, so each sample is 0 (answer A) or 1 (answer B).
    return rng.choice(len(pi), size = G, p=pi)
    

def group_advantages(rewards, eps=1e-8):
    # TODO: (reward − group mean) / (group std + eps)
    group_mean = np.mean(rewards)
    group_std = np.std(rewards)

    return (rewards - group_mean)/ (group_std + eps)

    ...

def grad_logits(samples, adv, pi):
    # TODO: average over the group of  adv_i × (one_hot(y_i) − pi)
    # Hint: np.eye(2)[y] is the one-hot vector for answer y
     # samples: which answer each sample was, e.g. [0, 1, 1]       shape (G,)
    # adv:     each sample's advantage,     e.g. [1.15, -0.58, -0.58]  shape (G,)
    # pi:      current probabilities,       e.g. [0.3, 0.7]       shape (2,)
    one_hot = np.eye(len(pi))[samples]
    directions = one_hot - pi 
    push = adv[:,None]*directions
    return push.mean(axis=0)

    


# ---- tests: all must pass before you train ----
assert np.allclose(softmax(np.array([0.0, 0.0])), [0.5, 0.5])
assert np.allclose(group_advantages(np.array([10.0, 2, 2, 2])),
                   [1.732, -0.577, -0.577, -0.577], atol=1e-3)
assert np.allclose(group_advantages(np.array([10.0, 10, 10, 10])), [0, 0, 0, 0])
assert np.allclose(grad_logits(np.array([0]), np.array([1.0]), np.array([0.3, 0.7])),
                   [0.7, -0.7])
pi_test = np.array([0.2, 0.5, 0.3])
samples_test = np.array([1, 2, 1, 0])
adv_test = group_advantages(np.array([1.0, 0.0, 1.0, 0.0]))
assert np.allclose(adv_test, [1, -1, 1, -1])
assert np.allclose(grad_logits(samples_test, adv_test, pi_test), [-0.25, 0.5, -0.25])
print("all tests pass")


# ---- training ----
z = np.log(np.array([0.3, 0.7]))     # TODO: logits that give pi = [0.3, 0.7]. Hint: log of the probabilities
lr, G, steps = 0.1, 8, 200
history = []

for step in range(steps):
    pi = softmax(z)
    history.append(pi[0])
    # TODO: sample a group, look up its rewards, compute advantages,
    #       compute the gradient, and update z (ascent: z = z + lr * grad)
    samples = sample_group(pi, G)
    rewards = REWARDS[samples]
    advantage = group_advantages(rewards)
    grad = grad_logits(samples, advantage, pi)

    z = z + lr*grad

plt.plot(history)
plt.xlabel("training step")
plt.ylabel("probability of answer A")
plt.title("GRPO on the two-answer model")
plt.savefig("day23/pi_A.png")