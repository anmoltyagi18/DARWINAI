"""
Market Regime Detection using K-Means Clustering + Hidden Markov Model
Regimes: Bull, Bear, Sideways, High Volatility
Features: price returns, volume, volatility indicators (ATR, rolling std, RSI)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1.  SYNTHETIC MARKET DATA GENERATOR
# ─────────────────────────────────────────────
def generate_market_data(n=2000, seed=42):
    """Generate realistic synthetic OHLCV data with known regime periods."""
    rng = np.random.default_rng(seed)

    # True regime sequence
    regime_lengths = []
    regime_labels  = []
    regime_map = {0: 'Bull', 1: 'Bear', 2: 'Sideways', 3: 'High Volatility'}

    regime_params = {
        # (daily_drift, daily_vol, volume_mean, volume_std)
        0: ( 0.0008, 0.010,  1.2e6, 2e5),   # Bull
        1: (-0.0010, 0.013,  1.4e6, 3e5),   # Bear
        2: ( 0.0001, 0.005,  0.8e6, 1e5),   # Sideways
        3: (-0.0002, 0.025,  2.0e6, 5e5),   # High Volatility
    }

    # Transition matrix (rows = from, cols = to)
    T = np.array([
        [0.97, 0.01, 0.01, 0.01],
        [0.01, 0.96, 0.02, 0.01],
        [0.02, 0.01, 0.95, 0.02],
        [0.02, 0.02, 0.02, 0.94],
    ])

    # Simulate regime sequence via MC
    true_regimes = np.zeros(n, dtype=int)
    true_regimes[0] = 0
    for t in range(1, n):
        true_regimes[t] = rng.choice(4, p=T[true_regimes[t-1]])

    # Simulate price & volume
    prices  = np.zeros(n)
    volumes = np.zeros(n)
    prices[0] = 100.0

    for t in range(1, n):
        r = true_regimes[t]
        drift, vol, vmean, vstd = regime_params[r]
        ret = drift + vol * rng.standard_normal()
        prices[t]  = prices[t-1] * np.exp(ret)
        volumes[t] = max(1000, vmean + vstd * rng.standard_normal())

    volumes[0] = volumes[1]

    df = pd.DataFrame({
        'close': prices,
        'volume': volumes,
        'true_regime': true_regimes,
        'true_regime_name': [regime_map[r] for r in true_regimes],
    })
    df.index = pd.date_range('2018-01-01', periods=n, freq='B')
    return df


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_features(df, atr_window=14, vol_window=20, rsi_window=14,
                     ma_short=10, ma_long=50):
    df = df.copy()

    # Log returns
    df['log_return']   = np.log(df['close'] / df['close'].shift(1))
    df['return_5d']    = df['log_return'].rolling(5).mean()
    df['return_20d']   = df['log_return'].rolling(20).mean()

    # Rolling volatility (close-to-close)
    df['vol_20']       = df['log_return'].rolling(vol_window).std() * np.sqrt(252)
    df['vol_5']        = df['log_return'].rolling(5).std() * np.sqrt(252)

    # ATR (using only close since we have no high/low; approximate)
    df['tr']           = df['close'].diff().abs()
    df['atr']          = df['tr'].rolling(atr_window).mean()

    # Volume features
    df['vol_ratio']    = df['volume'] / df['volume'].rolling(20).mean()
    df['log_volume']   = np.log(df['volume'])

    # RSI
    delta   = df['close'].diff()
    gain    = delta.clip(lower=0).rolling(rsi_window).mean()
    loss    = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs      = gain / (loss + 1e-9)
    df['rsi'] = 100 - 100 / (1 + rs)

    # Trend: price vs MAs
    df['ma_short']     = df['close'].rolling(ma_short).mean()
    df['ma_long']      = df['close'].rolling(ma_long).mean()
    df['trend_signal'] = (df['ma_short'] - df['ma_long']) / df['ma_long']

    # Price momentum
    df['momentum_10']  = df['close'].pct_change(10)
    df['momentum_20']  = df['close'].pct_change(20)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    'log_return', 'return_5d', 'return_20d',
    'vol_20', 'vol_5', 'atr',
    'vol_ratio', 'log_volume',
    'rsi', 'trend_signal',
    'momentum_10', 'momentum_20',
]


# ─────────────────────────────────────────────
# 3.  K-MEANS CLUSTER INITIALISATION
# ─────────────────────────────────────────────
def fit_kmeans(X_scaled, n_clusters=4, seed=42):
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(X_scaled)
    return km


def map_clusters_to_regimes(df, cluster_labels):
    """
    Map numeric cluster IDs to regime names using domain heuristics:
    - Bull:            high return, low vol
    - Bear:            low/neg return, moderate-high vol
    - Sideways:        near-zero return, low vol, low volume ratio
    - High Volatility: highest vol_20, large |return|
    """
    df = df.copy()
    df['cluster'] = cluster_labels

    stats = df.groupby('cluster')[['return_20d', 'vol_20', 'vol_ratio', 'trend_signal']].mean()

    # Score each cluster
    regime_scores = {}
    for cid, row in stats.iterrows():
        scores = {
            'Bull':            row['return_20d'] * 10 + row['trend_signal'] * 5 - row['vol_20'],
            'Bear':           -row['return_20d'] * 10 - row['trend_signal'] * 5 - row['vol_20'] * 0.5,
            'High Volatility': row['vol_20'] * 10 + abs(row['return_20d']) * 3,
            'Sideways':       -abs(row['return_20d']) * 8 - row['vol_20'] * 5,
        }
        regime_scores[cid] = scores

    # Hungarian-style greedy assignment
    assigned = {}
    used_regimes = set()
    for _ in range(4):
        best_score = -np.inf
        best_cid = best_regime = None
        for cid, sc in regime_scores.items():
            if cid in assigned:
                continue
            for reg, val in sc.items():
                if reg in used_regimes:
                    continue
                if val > best_score:
                    best_score = val
                    best_cid = cid
                    best_regime = reg
        assigned[best_cid] = best_regime
        used_regimes.add(best_regime)

    df['regime_kmeans'] = df['cluster'].map(assigned)
    return df, assigned


# ─────────────────────────────────────────────
# 4.  GAUSSIAN HMM (SCRATCH IMPLEMENTATION)
# ─────────────────────────────────────────────
class GaussianHMM:
    """
    Full Gaussian HMM with:
      - Baum-Welch EM training
      - Viterbi decoding
      - Forward-Backward smoothing
    """

    def __init__(self, n_states=4, n_iter=100, tol=1e-4, seed=42):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.seed     = seed

    # ── initialise from K-Means ──────────────────────────
    def init_from_kmeans(self, X, cluster_labels):
        K, D = self.n_states, X.shape[1]
        rng  = np.random.default_rng(self.seed)

        # Start prob
        self.pi = np.full(K, 1.0 / K)

        # Transition matrix – nearly diagonal + small noise
        self.A = np.full((K, K), 0.02 / (K - 1))
        np.fill_diagonal(self.A, 0.96)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Emission params from cluster statistics
        self.mu    = np.zeros((K, D))
        self.sigma = np.zeros((K, D, D))
        for k in range(K):
            mask = cluster_labels == k
            pts  = X[mask] if mask.sum() > 1 else X
            self.mu[k]    = pts.mean(axis=0)
            cov           = np.cov(pts.T) if mask.sum() > 1 else np.eye(D)
            self.sigma[k] = cov + np.eye(D) * 1e-3

    # ── emission log-prob ────────────────────────────────
    def _log_emission(self, X):
        """Returns (T, K) log-prob matrix."""
        T, D = X.shape
        log_b = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            try:
                rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k],
                                         allow_singular=True)
                log_b[:, k] = rv.logpdf(X)
            except Exception:
                log_b[:, k] = -1e10
        return log_b

    # ── forward algorithm (log-scale) ───────────────────
    def _forward(self, log_b):
        T, K = log_b.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(self.pi + 1e-300) + log_b[0]
        log_A = np.log(self.A + 1e-300)

        for t in range(1, T):
            for k in range(K):
                vals = log_alpha[t-1] + log_A[:, k]
                log_alpha[t, k] = np.logaddexp.reduce(vals) + log_b[t, k]
        return log_alpha

    # ── backward algorithm (log-scale) ──────────────────
    def _backward(self, log_b):
        T, K = log_b.shape
        log_beta = np.zeros((T, K))
        log_A    = np.log(self.A + 1e-300)

        for t in range(T-2, -1, -1):
            for i in range(K):
                vals = log_A[i, :] + log_b[t+1, :] + log_beta[t+1, :]
                log_beta[t, i] = np.logaddexp.reduce(vals)
        return log_beta

    # ── Baum-Welch EM ────────────────────────────────────
    def fit(self, X):
        T, D = X.shape
        K    = self.n_states
        prev_loglik = -np.inf

        for iteration in range(self.n_iter):
            log_b     = self._log_emission(X)
            log_alpha = self._forward(log_b)
            log_beta  = self._backward(log_b)

            # Log-likelihood
            loglik = np.logaddexp.reduce(log_alpha[-1])

            # Posterior state probs  γ(t,k)
            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Transition posteriors  ξ(t,i,j)
            log_A = np.log(self.A + 1e-300)
            new_A = np.zeros((K, K))
            for t in range(T-1):
                for i in range(K):
                    for j in range(K):
                        new_A[i, j] += np.exp(
                            log_alpha[t, i] + log_A[i, j] +
                            log_b[t+1, j]  + log_beta[t+1, j] - loglik
                        )

            # M-step
            self.pi = gamma[0] / gamma[0].sum()
            self.pi = np.clip(self.pi, 1e-6, None)
            self.pi /= self.pi.sum()

            self.A = new_A + 1e-6
            self.A /= self.A.sum(axis=1, keepdims=True)

            for k in range(K):
                w = gamma[:, k]
                w_sum = w.sum() + 1e-9
                self.mu[k] = (w[:, None] * X).sum(axis=0) / w_sum
                diff = X - self.mu[k]
                self.sigma[k] = (w[:, None, None] * np.einsum('ti,tj->tij', diff, diff)
                                 ).sum(axis=0) / w_sum + np.eye(D) * 1e-3

            if abs(loglik - prev_loglik) < self.tol:
                print(f"  HMM converged at iteration {iteration+1}  (ΔlogL={loglik-prev_loglik:.6f})")
                break
            prev_loglik = loglik

        self.loglik_ = loglik
        return self

    # ── Viterbi decoding ─────────────────────────────────
    def predict(self, X):
        T, K = X.shape[0], self.n_states
        log_b = self._log_emission(X)
        log_A = np.log(self.A + 1e-300)

        viterbi   = np.full((T, K), -np.inf)
        backtrack = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(self.pi + 1e-300) + log_b[0]

        for t in range(1, T):
            for k in range(K):
                scores = viterbi[t-1] + log_A[:, k]
                best   = np.argmax(scores)
                viterbi[t, k]   = scores[best] + log_b[t, k]
                backtrack[t, k] = best

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(viterbi[-1])
        for t in range(T-2, -1, -1):
            path[t] = backtrack[t+1, path[t+1]]
        return path

    # ── Posterior state probabilities ───────────────────
    def predict_proba(self, X):
        log_b     = self._log_emission(X)
        log_alpha = self._forward(log_b)
        log_beta  = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)


# ─────────────────────────────────────────────
# 5.  REGIME MAPPING FOR HMM STATES
# ─────────────────────────────────────────────
def map_hmm_states(df, hmm_labels, feature_df, scaler):
    """Map HMM state IDs to regime names using emission statistics."""
    # Unscale means
    df = df.copy()
    df['hmm_state'] = hmm_labels

    feature_df = feature_df.copy()
    feature_df['hmm_state'] = hmm_labels

    stats = feature_df.groupby('hmm_state')[FEATURE_COLS].mean()

    ri = {c: i for i, c in enumerate(FEATURE_COLS)}
    assigned = {}
    used = set()

    for _ in range(4):
        best_score = -np.inf
        best_state = best_regime = None
        for sid, row in stats.iterrows():
            if sid in assigned:
                continue
            scores = {
                'Bull':            row['return_20d'] * 10 + row['trend_signal'] * 5 - row['vol_20'],
                'Bear':           -row['return_20d'] * 10 - row['trend_signal'] * 5 - row['vol_20'] * 0.5,
                'High Volatility': row['vol_20'] * 10 + abs(row['return_20d']) * 3,
                'Sideways':       -abs(row['return_20d']) * 8 - row['vol_20'] * 5,
            }
            for reg, val in scores.items():
                if reg in used:
                    continue
                if val > best_score:
                    best_score = val
                    best_state = sid
                    best_regime = reg
        assigned[best_state] = best_regime
        used.add(best_regime)

    df['regime_hmm'] = df['hmm_state'].map(assigned)
    return df, assigned


# ─────────────────────────────────────────────
# 6.  METRICS
# ─────────────────────────────────────────────
def regime_metrics(df):
    """Compute per-regime statistics."""
    rows = []
    for regime in ['Bull', 'Bear', 'Sideways', 'High Volatility']:
        mask  = df['regime_hmm'] == regime
        sub   = df[mask]
        if len(sub) == 0:
            continue
        ann_ret = sub['log_return'].mean() * 252
        ann_vol = sub['log_return'].std() * np.sqrt(252)
        sharpe  = ann_ret / (ann_vol + 1e-9)
        rows.append({
            'Regime':      regime,
            'Count':       len(sub),
            'Pct (%)':     round(len(sub) / len(df) * 100, 1),
            'Ann Ret (%)': round(ann_ret * 100, 2),
            'Ann Vol (%)': round(ann_vol * 100, 2),
            'Sharpe':      round(sharpe, 2),
            'Avg Vol/MA':  round(sub['vol_ratio'].mean(), 2),
            'Avg RSI':     round(sub['rsi'].mean(), 1),
        })
    return pd.DataFrame(rows).set_index('Regime')


# ─────────────────────────────────────────────
# 7.  VISUALISATION
# ─────────────────────────────────────────────
REGIME_COLORS = {
    'Bull':            '#2ecc71',
    'Bear':            '#e74c3c',
    'Sideways':        '#f39c12',
    'High Volatility': '#9b59b6',
}
REGIME_ORDER = ['Bull', 'Bear', 'Sideways', 'High Volatility']


def plot_full_dashboard(df, proba, metrics_df, scaler, feature_df, km_labels,
                        save_path='/mnt/user-data/outputs/market_regime_dashboard.png'):
    fig = plt.figure(figsize=(22, 28), facecolor='#0d1117')
    gs  = GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.35)

    txt_kw = dict(color='white')
    ax_bg  = '#161b22'
    grid_c = '#30363d'

    def style_ax(ax, title=''):
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(grid_c)
        ax.grid(True, color=grid_c, linewidth=0.5, alpha=0.7)
        if title:
            ax.set_title(title, color='#e6edf3', fontsize=10, fontweight='bold', pad=6)

    # ── Title ──────────────────────────────────
    fig.text(0.5, 0.975, 'Market Regime Detection · K-Means + Hidden Markov Model',
             ha='center', va='top', color='#e6edf3', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.963, 'Features: Price Returns · Volume · Volatility (ATR, Rolling Std, RSI)',
             ha='center', va='top', color='#8b949e', fontsize=10)

    # ── [Row 0] Price + HMM Regimes ────────────
    ax0 = fig.add_subplot(gs[0, :])
    style_ax(ax0, 'Price Series with HMM-Detected Regimes')
    ax0.plot(df.index, df['close'], color='#58a6ff', linewidth=0.9, zorder=3)

    prev_regime = None
    start_idx   = df.index[0]
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['regime_hmm'] != prev_regime:
            if prev_regime is not None:
                ax0.axvspan(start_idx, idx,
                            alpha=0.25, color=REGIME_COLORS.get(prev_regime, 'grey'), zorder=1)
            start_idx   = idx
            prev_regime = row['regime_hmm']
    ax0.axvspan(start_idx, df.index[-1], alpha=0.25,
                color=REGIME_COLORS.get(prev_regime, 'grey'), zorder=1)

    legend_patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.6, label=r)
                      for r in REGIME_ORDER]
    ax0.legend(handles=legend_patches, loc='upper left', fontsize=8,
               facecolor=ax_bg, edgecolor=grid_c,
               labelcolor='white', framealpha=0.9)
    ax0.set_ylabel('Price', **txt_kw, fontsize=8)

    # ── [Row 1] Regime probability bands ───────
    ax1 = fig.add_subplot(gs[1, :])
    style_ax(ax1, 'HMM State Posterior Probabilities')
    # proba columns are ordered by HMM state; we stack them
    bottom = np.zeros(len(df))
    state_labels = [None] * 4
    # Use the hmm_state→regime mapping embedded in df
    state_to_regime = dict(zip(df['hmm_state'], df['regime_hmm']))
    for k in range(4):
        regime_name = state_to_regime.get(k, f'State {k}')
        ax1.fill_between(df.index, bottom, bottom + proba[:, k],
                         alpha=0.75, color=REGIME_COLORS.get(regime_name, 'grey'),
                         label=regime_name)
        bottom += proba[:, k]
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('P(Regime)', **txt_kw, fontsize=8)
    ax1.legend(loc='upper right', fontsize=7, facecolor=ax_bg,
               edgecolor=grid_c, labelcolor='white', ncol=4, framealpha=0.9)

    # ── [Row 2] Volatility + Volume ─────────────
    ax2a = fig.add_subplot(gs[2, :2])
    style_ax(ax2a, '20-Day Annualised Volatility')
    ax2a.plot(df.index, df['vol_20'] * 100, color='#ffa657', linewidth=0.9)
    ax2a.set_ylabel('Vol (%)', **txt_kw, fontsize=8)

    ax2b = fig.add_subplot(gs[2, 2])
    style_ax(ax2b, 'Volume Ratio (vs 20-day MA)')
    ax2b.plot(df.index, df['vol_ratio'], color='#79c0ff', linewidth=0.9)
    ax2b.axhline(1.0, color='#8b949e', linestyle='--', linewidth=0.8)
    ax2b.set_ylabel('Ratio', **txt_kw, fontsize=8)

    # ── [Row 3] RSI + Trend Signal ──────────────
    ax3a = fig.add_subplot(gs[3, :2])
    style_ax(ax3a, 'RSI (14-day)')
    ax3a.plot(df.index, df['rsi'], color='#d2a8ff', linewidth=0.9)
    ax3a.axhline(70, color='#e74c3c', linestyle='--', linewidth=0.7)
    ax3a.axhline(30, color='#2ecc71', linestyle='--', linewidth=0.7)
    ax3a.set_ylim(0, 100)
    ax3a.set_ylabel('RSI', **txt_kw, fontsize=8)

    ax3b = fig.add_subplot(gs[3, 2])
    style_ax(ax3b, 'Trend Signal (MA crossover)')
    ax3b.plot(df.index, df['trend_signal'] * 100, color='#56d364', linewidth=0.9)
    ax3b.axhline(0, color='#8b949e', linestyle='--', linewidth=0.7)
    ax3b.set_ylabel('Signal (%)', **txt_kw, fontsize=8)

    # ── [Row 4] PCA scatter (K-Means) + PCA scatter (HMM) ──
    scaler2 = StandardScaler()
    X_pca   = scaler2.fit_transform(feature_df[FEATURE_COLS].values)
    pca     = PCA(n_components=2)
    Z       = pca.fit_transform(X_pca)
    exp_var = pca.explained_variance_ratio_

    ax4a = fig.add_subplot(gs[4, 0])
    style_ax(ax4a, 'PCA: K-Means Clusters')
    for r in REGIME_ORDER:
        mask = feature_df['regime_kmeans'] == r
        ax4a.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.4,
                     color=REGIME_COLORS[r], label=r)
    ax4a.set_xlabel(f'PC1 ({exp_var[0]*100:.1f}%)', color='#8b949e', fontsize=7)
    ax4a.set_ylabel(f'PC2 ({exp_var[1]*100:.1f}%)', color='#8b949e', fontsize=7)
    ax4a.legend(fontsize=6, facecolor=ax_bg, edgecolor=grid_c,
                labelcolor='white', markerscale=3)

    ax4b = fig.add_subplot(gs[4, 1])
    style_ax(ax4b, 'PCA: HMM States')
    for r in REGIME_ORDER:
        mask = feature_df['regime_hmm'] == r
        ax4b.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.4,
                     color=REGIME_COLORS[r], label=r)
    ax4b.set_xlabel(f'PC1 ({exp_var[0]*100:.1f}%)', color='#8b949e', fontsize=7)
    ax4b.set_ylabel(f'PC2 ({exp_var[1]*100:.1f}%)', color='#8b949e', fontsize=7)
    ax4b.legend(fontsize=6, facecolor=ax_bg, edgecolor=grid_c,
                labelcolor='white', markerscale=3)

    # ── [Row 4, col 2] HMM Transition Matrix ───
    ax4c = fig.add_subplot(gs[4, 2])
    style_ax(ax4c, 'HMM Transition Matrix')
    import matplotlib.colors as mcolors
    cmap = plt.cm.Blues
    im   = ax4c.imshow(hmm_model.A, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax4c, fraction=0.046)
    regime_short = ['Bull', 'Bear', 'Side', 'HiVol']
    # Build ordered state labels matching assigned mapping
    state_order_labels = [state_to_regime.get(k, f'S{k}')[:4] for k in range(4)]
    ax4c.set_xticks(range(4)); ax4c.set_xticklabels(state_order_labels, color='#8b949e', fontsize=7)
    ax4c.set_yticks(range(4)); ax4c.set_yticklabels(state_order_labels, color='#8b949e', fontsize=7)
    for i in range(4):
        for j in range(4):
            ax4c.text(j, i, f'{hmm_model.A[i,j]:.2f}', ha='center', va='center',
                      color='white' if hmm_model.A[i,j] < 0.6 else '#0d1117', fontsize=7)

    # ── [Row 5] Metrics table ────────────────────
    ax5 = fig.add_subplot(gs[5, :])
    ax5.set_facecolor(ax_bg)
    ax5.axis('off')
    ax5.set_title('Regime Statistics (HMM)', color='#e6edf3', fontsize=10,
                  fontweight='bold', pad=6)

    col_labels = metrics_df.reset_index().columns.tolist()
    cell_data  = metrics_df.reset_index().values.tolist()
    col_widths = [0.12, 0.08, 0.08, 0.12, 0.12, 0.10, 0.12, 0.10]

    tbl = ax5.table(
        cellText  = cell_data,
        colLabels = col_labels,
        cellLoc   = 'center',
        loc       = 'center',
        colWidths = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#21262d' if r % 2 == 0 else ax_bg)
        cell.set_text_props(color='white')
        cell.set_edgecolor(grid_c)
        if r == 0:
            cell.set_facecolor('#30363d')
            cell.set_text_props(color='#e6edf3', fontweight='bold')
        # Colour regime name cell
        if r > 0 and c == 0:
            regime_name = cell_data[r-1][0]
            cell.set_facecolor(REGIME_COLORS.get(regime_name, ax_bg))
            cell.set_text_props(color='white', fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"  Dashboard saved → {save_path}")


# ─────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  MARKET REGIME DETECTION PIPELINE")
    print("=" * 60)

    # Step 1 – Data
    print("\n[1/6] Generating synthetic market data …")
    raw_df = generate_market_data(n=2000)
    print(f"      {len(raw_df)} trading days generated")

    # Step 2 – Features
    print("\n[2/6] Engineering features …")
    feat_df = compute_features(raw_df)
    print(f"      {len(feat_df)} usable rows after windowing  |  {len(FEATURE_COLS)} features")

    # Step 3 – Scale
    print("\n[3/6] Scaling features …")
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df[FEATURE_COLS].values)

    # Step 4 – K-Means
    print("\n[4/6] Fitting K-Means (k=4) …")
    km        = fit_kmeans(X_scaled, n_clusters=4)
    feat_df, km_map = map_clusters_to_regimes(feat_df, km.labels_)
    print(f"      Cluster → Regime mapping: {km_map}")
    print(f"      Inertia: {km.inertia_:.2f}")

    # Step 5 – HMM
    print("\n[5/6] Fitting Gaussian HMM (Baum-Welch, 4 states) …")
    hmm_model = GaussianHMM(n_states=4, n_iter=80, tol=1e-5, seed=42)
    hmm_model.init_from_kmeans(X_scaled, km.labels_)
    hmm_model.fit(X_scaled)
    print(f"      Final log-likelihood: {hmm_model.loglik_:.2f}")

    hmm_labels = hmm_model.predict(X_scaled)
    proba      = hmm_model.predict_proba(X_scaled)

    feat_df, hmm_map = map_hmm_states(feat_df, hmm_labels, feat_df, scaler)
    print(f"      State → Regime mapping: {hmm_map}")

    # Align index with raw_df for plotting
    plot_df = feat_df.copy()

    # Step 6 – Metrics + Plot
    print("\n[6/6] Computing metrics & rendering dashboard …")
    metrics = regime_metrics(plot_df)
    print("\n  Regime Statistics:")
    print(metrics.to_string())

    import os
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    plot_full_dashboard(plot_df, proba, metrics, scaler, feat_df, km.labels_)

    # Also save the metrics CSV
    metrics.to_csv('/mnt/user-data/outputs/regime_metrics.csv')
    print("  Metrics CSV saved → /mnt/user-data/outputs/regime_metrics.csv")

    print("\n✓ Pipeline complete.")
