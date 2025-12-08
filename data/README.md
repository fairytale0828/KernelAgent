# Long-Term Strategy State Storage

This directory contains **persistent policy data** that should be preserved across sessions.

## Important

⚠️ **DO NOT DELETE** files in this directory unless you want to reset the learning progress.

Unlike `triton_kernel_logs/` which contains temporary session logs, this directory stores:
- Long-term strategy performance statistics
- Prior knowledge and learned experience
- Global iteration counters

## Files

### `policy_state.json`

The main strategy state file containing:

```json
{
  "stats": {
    "op_type": {
      "strategy_id": {
        "n": 10,           // Number of attempts
        "mu_data": 1.25    // Average speedup from data
      }
    }
  },
  "priors": {
    "op_type": {
      "strategy_id": 1.2   // Prior expected speedup
    }
  },
  "global_iteration": 42,  // Total iterations across all sessions
  "metadata": {
    "prior_weight": 5.0,
    "exploration_coef": 2.0
  }
}
```

### Strategy Selection Algorithm

The system uses a **Bayesian Bandit with UCB** approach:

1. **Blended Mean** (exploitation):
   ```
   BlendedMean = (n * mu_data + n0 * mu_prior) / (n + n0)
   ```
   - Combines prior knowledge with observed data
   - Early on: relies more on priors
   - Later: relies more on actual performance data

2. **Exploration Bonus** (exploration):
   ```
   Bonus = c * sqrt(log(T) / (n + 1))
   ```
   - Encourages trying less-explored strategies
   - Decreases as strategy is tried more

3. **UCB Score**:
   ```
   UCB = BlendedMean + Bonus
   ```
   - Strategies with highest UCB are selected
   - Balances exploitation vs exploration

## Backup and Recovery

To backup your learning progress:
```bash
cp data/policy_state.json data/policy_state.backup.json
```

To reset and start fresh:
```bash
rm data/policy_state.json
# System will initialize with priors on next run
```

## Migration from Old Format

If you have an old `triton_kernel_logs/strategy_scores.json`, run:
```bash
python scripts/migrate_strategy_data.py
```

This will convert the old format to the new Bayesian format with priors.
