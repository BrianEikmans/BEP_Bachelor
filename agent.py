import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import config
from scoring import calculate_score, get_hand_rank

# Hand type index for one-hot encoding
HAND_TYPES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
]
HAND_TYPE_INDEX = {h: i for i, h in enumerate(HAND_TYPES)}
NUM_HAND_TYPES  = len(HAND_TYPES)

# Action feature size:
#   8  (card selection mask)
#   12 (hand type one-hot — of KEPT cards for discards, selected cards for plays)
#   1  (normalised chip total — of kept/selected cards)
#   1  (signed card count: +n/8 for plays, -n/8 for discards — encodes action type)
ACTION_FEATURE_SIZE = config.HAND_SIZE + NUM_HAND_TYPES + 2


def encode_action(card_indices: list, hand: list, is_discard: bool = False) -> list:
    """
    Encode a card selection as a rich feature vector.

    For play actions (is_discard=False):
        Features describe the selected (played) cards.
    For discard actions (is_discard=True):
        Features describe the KEPT cards (what the agent is building toward).
        card_count is negative to signal a discard action.

    Features:
        [0:8]   binary card selection mask (which cards are selected)
        [8:20]  one-hot hand type (12 classes)
        [20]    chip value, normalised by max possible (5*11=55)
        [21]    card count: +n/HAND_SIZE for plays, -n/HAND_SIZE for discards
    """
    # Binary selection mask (always refers to selected/discarded cards)
    mask = [0.0] * config.HAND_SIZE
    for idx in card_indices:
        mask[idx] = 1.0

    if is_discard:
        # Features based on the KEPT cards — what remains after the discard
        kept_indices = [i for i in range(len(hand)) if i not in card_indices]
        feature_cards = [hand[i] for i in kept_indices]
    else:
        feature_cards = [hand[i] for i in card_indices]

    # Hand type one-hot
    hand_rank   = get_hand_rank(feature_cards) if feature_cards else "High Card"
    hand_onehot = [0.0] * NUM_HAND_TYPES
    hand_onehot[HAND_TYPE_INDEX[hand_rank]] = 1.0

    # Chip total
    chip_total = sum(c[2] for c in feature_cards) / 55.0

    # Signed card count: positive = play, negative = discard
    card_count = len(card_indices) / config.HAND_SIZE
    if is_discard:
        card_count = -card_count

    return mask + hand_onehot + [chip_total, card_count]


def encode_actions(valid_actions: list, hand: list) -> torch.Tensor:
    """
    Encode all valid actions into a (MAX_ACTIONS, ACTION_FEATURE_SIZE) tensor.
    Each action is a (card_indices, is_discard) tuple.
    Rows beyond len(valid_actions) are zero-padded.
    """
    n        = ReplayBuffer.MAX_ACTIONS
    features = torch.zeros(n, ACTION_FEATURE_SIZE, dtype=torch.float32)
    for i, (card_indices, is_discard) in enumerate(valid_actions):
        features[i] = torch.tensor(
            encode_action(card_indices, hand, is_discard), dtype=torch.float32
        )
    return features


import numpy as np

# =============================================================================
# SumTree  --  O(log n) priority sampling for PER
# =============================================================================

class SumTree:
    """Binary tree: leaves = priorities, internal nodes = subtree sums."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self.data     = [None] * capacity
        self.size     = 0
        self.ptr      = 0

    def add(self, priority: float, data) -> None:
        self.data[self.ptr] = data
        self.update(self.ptr + self.capacity, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        idx = leaf_idx // 2
        while idx >= 1:
            self.tree[idx] += delta
            idx //= 2

    def get(self, s: float) -> tuple:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            idx  = left if s <= self.tree[left] else left + 1
            if idx >= 2 * self.capacity:
                break
            if idx != left:
                s -= self.tree[left]
        data_idx = idx - self.capacity
        return idx, float(self.tree[idx]), self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[1])

    @property
    def max_priority(self) -> float:
        raw = float(self.tree[self.capacity : self.capacity + max(self.size, 1)].max())
        # Guard: if all priorities are 0 (fresh buffer), return 1.0 so the first
        # push gets a real priority and sample() never divides by zero → NaN.
        return raw if raw > 0.0 else 1.0


# =============================================================================
# Prioritized Experience Replay
# =============================================================================

class PrioritizedReplayBuffer:
    """
    PER buffer. Transitions sampled by |TD error|^alpha; corrected with
    IS weights (beta annealed from PER_BETA_START -> 1.0 over PER_BETA_STEPS).
    """
    MAX_ACTIONS = 438

    def __init__(self, capacity: int = config.REPLAY_CAPACITY):
        self.tree     = SumTree(capacity)
        self.capacity = capacity
        self.alpha    = config.PER_ALPHA
        self._step    = 0

    def push(self, state, action_features, reward,
             next_state, next_action_features, done) -> None:
        p = self.tree.max_priority ** self.alpha
        self.tree.add(p, (state, action_features, reward,
                          next_state, next_action_features, done))

    def sample(self, batch_size: int) -> tuple:
        fraction = min(self._step / config.PER_BETA_STEPS, 1.0)
        beta     = config.PER_BETA_START + fraction * (config.PER_BETA_END - config.PER_BETA_START)
        self._step += 1

        segment    = self.tree.total / batch_size
        indices    = []
        priorities = []
        data_batch = []

        for i in range(batch_size):
            s   = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            while data is None:
                s   = random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            data_batch.append(data)

        probs   = np.array(priorities, dtype=np.float64) / self.tree.total
        probs   = np.clip(probs, 1e-8, None)
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        states, afs, rewards, nstates, nafs, dones = zip(*data_batch)
        return (
            torch.tensor(states,  dtype=torch.float32),
            torch.tensor(afs,     dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(nstates, dtype=torch.float32),
            torch.stack(list(nafs)),
            torch.tensor(dones,   dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices: list, td_errors) -> None:
        for idx, err in zip(indices, td_errors):
            self.tree.update(idx, (abs(float(err)) + 1e-6) ** self.alpha)

    def __len__(self) -> int:
        return self.tree.size


# =============================================================================
# N-Step Return Buffer
# =============================================================================

class NStepBuffer:
    """
    Accumulates n transitions then pushes the n-step return to the PER buffer.
    G_t = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1}
    Bootstrap term (gamma^n * V) handled by the DQN target computation.
    """
    def __init__(self, n: int, gamma: float, replay_buffer: PrioritizedReplayBuffer):
        self.n      = n
        self.gamma  = gamma
        self.buffer = deque()
        self.replay = replay_buffer

    def push(self, state, action_features, reward,
             next_state, next_action_features, done) -> None:
        self.buffer.append((state, action_features, reward,
                            next_state, next_action_features, done))
        if len(self.buffer) >= self.n:
            self._flush_oldest()
        if done:
            while self.buffer:
                self._flush_oldest()

    def _flush_oldest(self) -> None:
        if not self.buffer:
            return
        state0, af0, _, _, _, _ = self.buffer[0]
        g = 0.0
        final_ns = final_na = None
        final_done = False
        for i, (_, _, r, ns, na, d) in enumerate(self.buffer):
            g += (self.gamma ** i) * r
            final_ns, final_na, final_done = ns, na, d
            if d:
                break
        self.replay.push(state0, af0, g, final_ns, final_na, final_done)
        self.buffer.popleft()

    def update_gamma(self, gamma: float) -> None:
        self.gamma = gamma


# =============================================================================
# Replay Buffer (plain — kept for backward checkpoint compatibility)
# =============================================================================

class ReplayBuffer:
    """Uniform random replay buffer."""

    MAX_ACTIONS = 438

    def __init__(self, capacity: int = config.REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_features, reward, next_state, next_action_features, done):
        self.buffer.append((state, action_features, reward, next_state, next_action_features, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, action_feats, rewards, next_states, next_action_feats, dones = zip(*batch)
        return (
            torch.tensor(states,         dtype=torch.float32),
            torch.tensor(action_feats,   dtype=torch.float32),
            torch.tensor(rewards,        dtype=torch.float32),
            torch.tensor(next_states,    dtype=torch.float32),
            torch.stack(list(next_action_feats)),
            torch.tensor(dones,          dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Q-Network
# =============================================================================

class DQNNetwork(nn.Module):
    """
    State-action value network: (state || action_features) -> Q-value.

    action_features now includes:
        - card selection mask       (8)
        - hand type one-hot         (12)  ← explicit hand quality signal
        - normalised chip total     (1)   ← explicit chip signal
        - normalised card count     (1)   ← explicit size signal

    This gives the network direct hand quality information rather than
    forcing it to infer poker hand rankings from card positions alone.

    Architecture:
        Input  : state_size + ACTION_FEATURE_SIZE
        Hidden : 256 -> 128
        Output : 1
    """

    def __init__(self, state_size: int):
        super().__init__()
        input_size = state_size + ACTION_FEATURE_SIZE
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state           : (B, state_size)
            action_features : (B, ACTION_FEATURE_SIZE)
        Returns:
            q_value         : (B,)
        """
        x = torch.cat([state, action_features], dim=-1)
        return self.net(x).squeeze(-1)


# =============================================================================
# DQN Agent
# =============================================================================

class DQNAgent:
    """
    DQN agent using a state-action value network with rich action features.
    """

    def __init__(self, state_size: int = config.STATE_SIZE, device_override: str = None):
        self.state_size    = state_size
        self.epsilon       = config.EPSILON_START
        if device_override is not None:
            self.device = torch.device(device_override)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net    = DQNNetwork(state_size).to(self.device)
        self.target_net    = DQNNetwork(state_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.replay_buffer = PrioritizedReplayBuffer()
        self.n_step_buffer = NStepBuffer(config.N_STEP, config.GAMMA, self.replay_buffer)
        self.steps_done    = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: list, valid_actions: list, hand: list) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state         : current state vector
            valid_actions : list of card-index lists from env.valid_play_actions()
            hand          : current cards in hand (for action feature encoding)

        Returns:
            action_idx : int index into valid_actions
        """
        n_valid = len(valid_actions)

        if random.random() < self.epsilon:
            return random.randrange(n_valid)

        with torch.no_grad():
            state_t      = torch.tensor(state, dtype=torch.float32).to(self.device)
            state_rep    = state_t.unsqueeze(0).expand(n_valid, -1)

            action_feats = encode_actions(valid_actions, hand).to(self.device)[:n_valid]

            q_vals = self.policy_net(state_rep, action_feats)
            return int(q_vals.argmax().item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store(self, state, action_features, reward, next_state, next_action_features, done):
        """Push through n-step buffer into PER."""
        self.n_step_buffer.push(
            state, action_features, reward,
            next_state, next_action_features, done
        )

    def train_step(self):
        """
        Sample a PER mini-batch and perform one Double DQN gradient update.
        Returns loss (float) or None if buffer too small.
        """
        if len(self.replay_buffer) < config.BATCH_SIZE:
            return None

        (states, action_feats, rewards, next_states,
         next_action_feats, dones, weights, indices) = \
            self.replay_buffer.sample(config.BATCH_SIZE)

        B = states.shape[0]
        states            = states.to(self.device)
        action_feats      = action_feats.to(self.device)
        rewards           = rewards.to(self.device)
        next_states       = next_states.to(self.device)
        next_action_feats = next_action_feats.to(self.device)
        dones             = dones.to(self.device)
        weights           = weights.to(self.device)

        # Current Q-values
        q_current = self.policy_net(states, action_feats)   # (B,)

        # Double DQN target:
        # 1. policy_net selects the best next action
        # 2. target_net evaluates its Q-value
        with torch.no_grad():
            M = PrioritizedReplayBuffer.MAX_ACTIONS
            next_states_exp  = next_states.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
            next_feats_flat  = next_action_feats.reshape(B * M, ACTION_FEATURE_SIZE)
            valid_flag       = next_action_feats.sum(dim=-1) > 0   # (B, M)

            # Step 1: policy_net selects
            q_next_policy    = self.policy_net(next_states_exp, next_feats_flat).reshape(B, M)
            q_next_policy[~valid_flag] = float("-inf")
            best_actions     = q_next_policy.argmax(dim=1)          # (B,)

            # Step 2: target_net evaluates
            q_next_target    = self.target_net(next_states_exp, next_feats_flat).reshape(B, M)
            q_next_target[~valid_flag] = float("-inf")
            q_next           = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # gamma^n for n-step bootstrap
            gamma_n  = config.GAMMA ** config.N_STEP
            q_target = rewards + gamma_n * q_next * (1.0 - dones)

        # PER-weighted loss
        td_errors = (q_current - q_target).detach()
        loss = (weights * nn.functional.smooth_l1_loss(
                    q_current, q_target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        self.steps_done += 1
        return loss.item()

    def decay_epsilon(self):
        """Multiplicative epsilon decay — call once per episode."""
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)

    def sync_target_network(self):
        """Soft update: target = tau*policy + (1-tau)*target for stability."""
        tau = 0.01
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def update_gamma(self, gamma: float) -> None:
        """Update n-step buffer's discount factor (called when switching run phases)."""
        self.n_step_buffer.update_gamma(gamma)

    # ------------------------------------------------------------------
    # Weight sync for parallel workers
    # ------------------------------------------------------------------

    def get_weights(self) -> bytes:
        """Serialise policy_net weights to bytes for sending to worker processes."""
        import io
        buf = io.BytesIO()
        torch.save(self.policy_net.state_dict(), buf)
        return buf.getvalue()

    def set_weights(self, weights: bytes) -> None:
        """Load serialised policy_net weights received from the learner process."""
        import io
        state_dict = torch.load(io.BytesIO(weights),
                                map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "policy_state_dict":    self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon":              self.epsilon,
            "steps_done":           self.steps_done,
        }, path)
        print(f"[Agent] Saved checkpoint -> {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon    = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        print(f"[Agent] Loaded checkpoint <- {path}")