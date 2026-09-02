# Spectral analysis of a trained Go1 policy

`run_go1_rollouts.py` evaluates a completed locomotion checkpoint found on
Eagle on 64 reproducible, environment-sampled velocity-tracking tasks. It uses
the selected environment's current default configuration rather than the
checkpoint's saved training overrides and records raw actuator torque at the
default 50 Hz policy rate. The generated comparison uses the flat-terrain Go1,
Spot, and SilverBadger environments and the same task seed for all three.

From the repository root on an Eagle GPU node:

```bash
python paper/policy_spectral_analysis/run_go1_rollouts.py
```

The runner defaults to the Go1 checkpoint
`260727-baseline-400M-ar1em1-seed0/checkpoints/000417792000`. Select another
completed checkpoint with `--checkpoint` and its environment with `--env-name`.
Results default to the untracked `paper/policy_spectral_analysis/data/go1_default`
directory.

Then create the torque spectrum averaged over time windows, tasks, and joints,
and report penalty values:

```bash
python paper/policy_spectral_analysis/plot_torque_spectrogram.py \
  paper/policy_spectral_analysis/data/go1_default \
  paper/policy_spectral_analysis/data/spot_default \
  paper/policy_spectral_analysis/data/silver_badger_default \
  --labels Go1 Spot SilverBadger
```

The plot and its companion NPZ are written by default to the `figures/`
directory next to the plotting script.

A raw-torque companion with `_linear` appended to its filename shows the same
PSD directly in `(N m)^2/Hz` instead of decibels.

The same command also writes a second figure with `_max_normalized` appended
to its name. Every actuator torque is first divided by that actuator's maximum
absolute value over all active samples and tasks. Each robot's resulting PSD
is then divided by its integral over the full Nyquist range. Every curve
therefore integrates to one, and the y-axis is the fraction of normalized
torque variance per hertz. Its companion NPZ includes the maximum used for
every actuator, the unit-area spectrum, and the per-task normalized penalties.

The analysis prints the task mean and standard deviation of the per-step cost,
summed over the 12 actuators, for torque rate and TFR with cutoff 5 Hz,
Butterworth order 1, and difference order 1. Both costs are divided by the
mean of their squared frequency response on a 16,384-point grid over
`[0, Nyquist)`. Thus their average frequency weight is one and their scales
are directly comparable for a white spectrum. TFR uses raw torque (N m), not
actuator-capacity normalization. The figure's companion NPZ stores the PSD and
per-task penalty values for later paper analysis.
