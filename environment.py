# environment.py — Portfolio Simulation Environment for P2-ETF-ERL-ENGINE
# Simulates portfolio dynamics for DDPG training and ERL episodes.
# Handles transaction costs, position sizing, reward computation,
# and experience replay buffer.
#
# Used by: ddpg_train.py, erl_train.py, predict.py

import os
import sys
import numpy as np
import pandas as pd
from collections import deque
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience replay buffer for DDPG.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = cfg.DDPG_BUFFER_SIZE):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            np.array(action,     dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        """True when buffer has enough samples for a training batch."""
        return len(self.buffer) >= cfg.DDPG_BATCH_SIZE


# ── Ornstein-Uhlenbeck Noise ───────────────────────────────────────────────────

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    Better than Gaussian noise for continuous action spaces — noise is
    mean-reverting so exploration doesn't drift indefinitely.
    """

    def __init__(
        self,
        size:  int,
        theta: float = cfg.DDPG_OU_THETA,
        sigma: float = cfg.DDPG_OU_SIGMA,
        mu:    float = 0.0,
    ):
        self.size  = size
        self.theta = theta
        self.sigma = sigma
        self.mu    = mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        dx = (
            self.theta * (self.mu - self.state) +
            self.sigma * np.random.randn(self.size)
        )
        self.state = self.state + dx
        return self.state.copy()

    def decay(self, factor: float = 0.995):
        """Decay sigma over training to reduce exploration over time."""
        self.sigma = max(self.sigma * factor, 0.01)


# ── Portfolio Environment ──────────────────────────────────────────────────────

class PortfolioEnv:
    """
    Portfolio simulation environment for DDPG training.

    State:  92-dim vector = TFT embedding (64) + HMM probs (8)
                          + current weights (19) + rolling Sharpe (1)
    Action: 19-dim softmax weights over [18 ETFs + CASH]
    Reward: log return vs benchmark - transaction cost

    Episode: one full trading period (e.g. one year of daily steps)
    """

    def __init__(
        self,
        embeddings:      pd.DataFrame,    # TFT embeddings (T, 64)
        hmm_probs:       pd.DataFrame,    # HMM probabilities (T, 8)
        asset_returns:   pd.DataFrame,    # ETF daily returns (T, N_ASSETS-1)  (excluding CASH)
        bench_returns:   pd.Series,       # AGG daily returns (T,)
        initial_capital: float = cfg.INITIAL_CAPITAL,
    ):
        # Align all inputs to common dates
        common_idx = (
            embeddings.index
            .intersection(hmm_probs.index)
            .intersection(asset_returns.index)
            .intersection(bench_returns.index)
        )
        common_idx = common_idx.sort_values()

        self.embeddings    = embeddings.reindex(common_idx).values.astype(np.float32)
        self.hmm_probs     = hmm_probs.reindex(common_idx).values.astype(np.float32)
        self.asset_returns = asset_returns.reindex(common_idx).values.astype(np.float32)
        self.bench_returns = bench_returns.reindex(common_idx).values.astype(np.float32)
        self.dates         = common_idx
        self.T             = len(common_idx)

        self.initial_capital = initial_capital
        self.noise           = OUNoise(cfg.N_ASSETS)

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode."""
        self.t               = 0
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [self.initial_capital]
        self.daily_returns   = []
        self.weights         = np.ones(cfg.N_ASSETS) / cfg.N_ASSETS
        self.noise.reset()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Assemble 92-dim state vector for current timestep.
        """
        t = min(self.t, self.T - 1)

        embedding       = self.embeddings[t]          # (64,)
        hmm_p           = self.hmm_probs[t]           # (8,)
        current_weights = self.weights                # (N_ASSETS,)
        sharpe          = self._rolling_sharpe()      # scalar

        state = np.concatenate([
            embedding,
            hmm_p,
            current_weights,
            np.array([sharpe], dtype=np.float32),
        ])
        return state.astype(np.float32)

    def _rolling_sharpe(self, window: int = cfg.KELLY_SHARPE_WINDOW) -> float:
        """Compute rolling Sharpe from recent daily returns."""
        if len(self.daily_returns) < 2:
            return 0.0
        recent = np.array(self.daily_returns[-window:])
        mean_r = recent.mean()
        std_r  = recent.std()
        return float((mean_r / (std_r + 1e-8)) * np.sqrt(252))

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute one trading step.

        Parameters
        ----------
        action : np.ndarray shape (N_ASSETS,)
            Raw policy output — will be softmax-normalised internally.

        Returns
        -------
        next_state : np.ndarray (92,)
        reward     : float
        done       : bool
        info       : dict
        """
        if self.t >= self.T - 1:
            return self._get_state(), 0.0, True, self._get_info(0.0, 0.0)

        # ── Normalise action to valid portfolio weights ─────────────────────
        weights = self._softmax(action)

        # ── Transaction cost ───────────────────────────────────────────────
        turnover = np.abs(weights - self.weights).sum()
        tx_cost  = turnover * cfg.TRANSACTION_COST

        # ── Compute portfolio return for this step ─────────────────────────
        # CASH earns 0 (conservative — no risk-free rate)
        etf_ret  = self.asset_returns[self.t]               # (N_ASSETS-1,)
        cash_ret = np.array([0.0], dtype=np.float32)
        all_ret  = np.concatenate([etf_ret, cash_ret])      # (N_ASSETS,)

        port_ret  = float(np.dot(weights, all_ret))
        net_ret   = port_ret - tx_cost
        bench_ret = float(self.bench_returns[self.t])

        # ── Reward: log excess return vs benchmark ─────────────────────────
        excess = net_ret - bench_ret
        reward = float(np.log1p(net_ret) - np.log1p(bench_ret + 1e-8))

        # ── Update portfolio value ─────────────────────────────────────────
        self.portfolio_value *= (1 + net_ret)
        self.portfolio_history.append(self.portfolio_value)
        self.daily_returns.append(net_ret)
        self.weights = weights

        # ── Advance time ───────────────────────────────────────────────────
        self.t += 1
        done    = self.t >= self.T - 1

        next_state = self._get_state()
        info       = self._get_info(net_ret, bench_ret)

        return next_state, reward, done, info

    def _get_info(self, net_ret: float, bench_ret: float) -> dict:
        return {
            'portfolio_value': self.portfolio_value,
            'net_return':      net_ret,
            'bench_return':    bench_ret,
            'excess_return':   net_ret - bench_ret,
            'weights':         self.weights.copy(),
            'rolling_sharpe':  self._rolling_sharpe(),
            'date':            self.dates[min(self.t, self.T-1)],
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Stable softmax — ensures weights sum to 1 and are all positive."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-8)

    def total_return(self) -> float:
        return self.portfolio_value / self.initial_capital - 1

    def sharpe_ratio(self) -> float:
        if len(self.daily_returns) < 2:
            return 0.0
        r    = np.array(self.daily_returns)
        mean = r.mean()
        std  = r.std()
        return float((mean / (std + 1e-8)) * np.sqrt(252))

    def max_drawdown(self) -> float:
        vals = np.array(self.portfolio_history)
        peak = np.maximum.accumulate(vals)
        dd   = (vals - peak) / (peak + 1e-8)
        return float(dd.min())

    def excess_return(self, bench_total_return: float) -> float:
        return self.total_return() - bench_total_return

    def summary(self) -> dict:
        return {
            'total_return':  self.total_return(),
            'sharpe_ratio':  self.sharpe_ratio(),
            'max_drawdown':  self.max_drawdown(),
            'n_steps':       self.t,
            'final_value':   self.portfolio_value,
        }


# ── ERL Episode ────────────────────────────────────────────────────────────────

class ERLEpisode:
    """
    Tracks a single ERL episode — stores first attempt, feedback,
    reflection, and second attempt for the ERL training loop.
    """

    def __init__(self, episode_id: int, regime: int, regime_name: str):
        self.episode_id   = episode_id
        self.regime       = regime
        self.regime_name  = regime_name

        # First attempt
        self.first_actions    = []
        self.first_rewards    = []
        self.first_states     = []
        self.first_return     = None
        self.first_excess     = None

        # Reflection
        self.reflection       = None
        self.reflection_text  = None

        # Second attempt
        self.second_actions   = []
        self.second_rewards   = []
        self.second_return    = None
        self.second_excess    = None

        # Metadata
        self.tft_attention    = None   # for reflection quality
        self.date_range       = None

    def record_first_attempt(
        self,
        actions: list,
        rewards: list,
        states:  list,
        total_return: float,
        excess_return: float,
        tft_attention: np.ndarray = None,
    ):
        self.first_actions   = actions
        self.first_rewards   = rewards
        self.first_states    = states
        self.first_return    = total_return
        self.first_excess    = excess_return
        self.tft_attention   = tft_attention

    def record_reflection(self, reflection_text: str):
        self.reflection_text = reflection_text

    def record_second_attempt(
        self,
        actions: list,
        rewards: list,
        total_return: float,
        excess_return: float,
    ):
        self.second_actions  = actions
        self.second_rewards  = rewards
        self.second_return   = total_return
        self.second_excess   = excess_return

    @property
    def improved(self) -> bool:
        """True if second attempt exceeded first attempt."""
        if self.second_excess is None or self.first_excess is None:
            return False
        return self.second_excess > self.first_excess

    @property
    def improvement(self) -> float:
        """Improvement in excess return from first to second attempt."""
        if self.second_excess is None or self.first_excess is None:
            return 0.0
        return self.second_excess - self.first_excess

    @property
    def worth_storing(self) -> bool:
        """True if improvement exceeds storage gate threshold."""
        return (
            self.improved and
            self.improvement >= cfg.ERL_MIN_EXCESS_TO_STORE
        )

    def to_dict(self) -> dict:
        return {
            'episode_id':     self.episode_id,
            'regime':         self.regime,
            'regime_name':    self.regime_name,
            'first_return':   self.first_return,
            'first_excess':   self.first_excess,
            'second_return':  self.second_return,
            'second_excess':  self.second_excess,
            'improvement':    self.improvement,
            'improved':       self.improved,
            'worth_storing':  self.worth_storing,
            'reflection':     self.reflection_text,
        }


# ── Benchmark Return Calculator ────────────────────────────────────────────────

def compute_benchmark_return(
    bench_returns: pd.Series,
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
) -> float:
    """
    Compute AGG total return over a given period.
    Used to compute excess return for ERL scoring.
    """
    mask   = (bench_returns.index >= start_date) & \
             (bench_returns.index <= end_date)
    period = bench_returns[mask]
    if len(period) == 0:
        return 0.0
    return float((1 + period).prod() - 1)


# ── Smoke Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[env] Running smoke test...")

    T = 252   # one year of daily steps

    # Dummy data
    np.random.seed(cfg.RANDOM_SEED)
    dates = pd.date_range('2023-01-01', periods=T, freq='B')

    embeddings    = pd.DataFrame(
        np.random.randn(T, cfg.TFT_EMBEDDING_DIM),
        index=dates,
        columns=[f'emb_{i}' for i in range(cfg.TFT_EMBEDDING_DIM)]
    )
    hmm_probs_df  = pd.DataFrame(
        np.random.dirichlet(np.ones(cfg.HMM_N_STATES), T),
        index=dates,
        columns=list(range(cfg.HMM_N_STATES))
    )
    # asset_returns has columns for all ETFs (excluding CASH)
    asset_returns = pd.DataFrame(
        np.random.randn(T, len(cfg.ASSETS)) * 0.01,
        index=dates,
        columns=cfg.ASSETS
    )
    bench_returns = pd.Series(
        np.random.randn(T) * 0.005,
        index=dates,
        name=cfg.BENCHMARK
    )

    env   = PortfolioEnv(embeddings, hmm_probs_df,
                         asset_returns, bench_returns)
    state = env.reset()
    assert state.shape == (cfg.DDPG_STATE_DIM,), \
        f"State shape mismatch: {state.shape}"

    # Run one full episode with random actions
    done = False
    step = 0
    while not done:
        action = np.random.randn(cfg.N_ASSETS)
        state, reward, done, info = env.step(action)
        step += 1

    summary = env.summary()
    print(f"[env] Episode complete: {step} steps")
    print(f"[env] Total return:  {summary['total_return']:.2%}")
    print(f"[env] Sharpe ratio:  {summary['sharpe_ratio']:.3f}")
    print(f"[env] Max drawdown:  {summary['max_drawdown']:.2%}")

    # Replay buffer test
    buf = ReplayBuffer(100)
    for _ in range(50):
        s  = np.random.randn(cfg.DDPG_STATE_DIM).astype(np.float32)
        a  = np.random.randn(cfg.N_ASSETS).astype(np.float32)
        r  = float(np.random.randn())
        ns = np.random.randn(cfg.DDPG_STATE_DIM).astype(np.float32)
        buf.push(s, a, r, ns, False)

    states, actions, rewards, next_states, dones = buf.sample(32)
    assert states.shape == (32, cfg.DDPG_STATE_DIM)
    assert actions.shape == (32, cfg.N_ASSETS)
    print(f"[env] Replay buffer: {len(buf)} samples, batch shape OK")

    # OU Noise test
    noise = OUNoise(cfg.N_ASSETS)
    samples = [noise.sample() for _ in range(100)]
    print(f"[env] OU noise mean: {np.mean(samples):.4f} (should be ~0)")

    # ERL Episode test
    ep = ERLEpisode(episode_id=1, regime=4, regime_name='Credit Stress')
    ep.record_first_attempt([], [], [], -0.02, -0.03)
    ep.record_reflection("In credit stress regime, reduce HYG exposure.")
    ep.record_second_attempt([], [], 0.01, 0.005)
    assert ep.improved
    assert ep.worth_storing
    print(f"[env] ERL episode: improved={ep.improved}, "
          f"improvement={ep.improvement:.3f}")

    print("\n✅ All environment tests passed")
