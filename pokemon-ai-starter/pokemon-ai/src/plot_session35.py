#!/usr/bin/env python
"""Generate Session 35 comprehensive Elo + eval trajectory plot."""
import json, re, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Load NEW Elo data ---
with open('data/eval/elo_session35_exp1.json') as f:
    elo_data = json.load(f)
with open('data/eval/eras.json') as f:
    eras = json.load(f)['eras']

elos = elo_data['elos']
cis = elo_data['cis']

bot_names = {'SH','Random','MaxBasePower','GreedySE','HazardSense','SwitchAwareEscape',
             'SetupThenSweep','SmartDmg','Tactical','Strategic'}

# Extract snapshot data
snap_iters, snap_elos, snap_lo, snap_hi, snap_names = [], [], [], [], []
for name in elos:
    if name in bot_names or name == 'BC_base':
        continue
    m = re.search(r'(\d{3,4})', name)
    if m:
        it = int(m.group(1))
        ci = cis[name]
        snap_iters.append(it)
        snap_elos.append(elos[name])
        snap_lo.append(ci['lo95'])
        snap_hi.append(ci['hi95'])
        snap_names.append(name)

# Add BC_base at iter 0
if 'BC_base' in elos:
    snap_iters.append(0)
    snap_elos.append(elos['BC_base'])
    ci = cis['BC_base']
    snap_lo.append(ci['lo95'])
    snap_hi.append(ci['hi95'])
    snap_names.append('BC_base')

order = np.argsort(snap_iters)
snap_iters = np.array(snap_iters)[order]
snap_elos = np.array(snap_elos)[order]
snap_lo = np.array(snap_lo)[order]
snap_hi = np.array(snap_hi)[order]
snap_names = np.array(snap_names)[order]

bot_elos = {name: elos[name] for name in bot_names if name in elos}

# --- Load eval data ---
eval_iters, eval_sh, eval_sd, eval_tac, eval_str, eval_savg = [], [], [], [], [], []
with open('data/eval/eval_history.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r['smart_avg']:
            eval_iters.append(int(r['iter']))
            eval_sh.append(float(r['SH']))
            eval_sd.append(float(r['SmartDmg']))
            eval_tac.append(float(r['Tactical']))
            eval_str.append(float(r['Strategic']))
            eval_savg.append(float(r['smart_avg']))

exp1_start = 1785
pre_mask = snap_iters < exp1_start
post_mask = snap_iters >= exp1_start

# --- Build figure ---
fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True,
                          gridspec_kw={'height_ratios': [1.4, 1, 1]})

# === Panel 1: Elo trajectory with eras ===
ax1 = axes[0]
for era in eras:
    lo, hi = era['iter_lo'], min(era['iter_hi'], 2100)
    ax1.axvspan(lo, hi, alpha=0.15, color=era['color'])
    mid = (lo + hi) / 2
    ax1.text(mid, 1085, era['id'], ha='center', va='bottom', fontsize=7, alpha=0.7)

ax1.fill_between(snap_iters[pre_mask], snap_lo[pre_mask], snap_hi[pre_mask], alpha=0.15, color='blue')
ax1.plot(snap_iters[pre_mask], snap_elos[pre_mask], 'o-', color='blue', markersize=5, label='Pre-Exp1 Elo', alpha=0.7)

ax1.fill_between(snap_iters[post_mask], snap_lo[post_mask], snap_hi[post_mask], alpha=0.2, color='green')
ax1.plot(snap_iters[post_mask], snap_elos[post_mask], 's-', color='green', markersize=7, linewidth=2.5, label='Exp 1 Elo (lam=0.95)')

for bname, style in [('Tactical', '--'), ('SH', '-'), ('SmartDmg', ':'), ('Strategic', '-.')]:
    if bname in bot_elos:
        ax1.axhline(y=bot_elos[bname], linestyle=style, alpha=0.4, color='gray',
                     label=f'{bname} ({bot_elos[bname]:.0f})')

ax1.annotate(f'sp1784\n(baseline)\n{elos.get("sp1784",0):.0f}',
             xy=(1784, elos.get('sp1784',0)), xytext=(1650, 955),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')
ax1.annotate(f'sp1984\n(Exp1 best)\n{elos.get("sp1984",0):.0f}',
             xy=(1984, elos.get('sp1984',0)), xytext=(1870, 1065),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=8, color='green', fontweight='bold')

ax1.axvline(x=exp1_start, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax1.set_ylabel('Elo Rating', fontsize=11)
ax1.set_title('Elo Trajectory with Training Eras - Session 35 Comprehensive Measurement', fontsize=13)
ax1.legend(loc='lower right', fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(680, 1100)

# === Panel 2: Per-bot win rates ===
ax2 = axes[1]
for era in eras:
    lo, hi = era['iter_lo'], min(era['iter_hi'], 2100)
    ax2.axvspan(lo, hi, alpha=0.1, color=era['color'])

ax2.plot(eval_iters, eval_sh, label='SH', alpha=0.7, linewidth=1)
ax2.plot(eval_iters, eval_sd, label='SmartDmg', alpha=0.7, linewidth=1)
ax2.plot(eval_iters, eval_tac, label='Tactical', alpha=0.7, linewidth=1)
ax2.plot(eval_iters, eval_str, label='Strategic', alpha=0.7, linewidth=1)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=exp1_start, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax2.set_ylabel('Win Rate %', fontsize=11)
ax2.set_title('Per-Bot Win Rate Over Training', fontsize=12)
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 80)

# === Panel 3: Smart average + Elo overlay ===
ax3 = axes[2]
for era in eras:
    lo, hi = era['iter_lo'], min(era['iter_hi'], 2100)
    ax3.axvspan(lo, hi, alpha=0.1, color=era['color'])

ax3.plot(eval_iters, eval_savg, alpha=0.3, color='blue', label='smart_avg (raw)')
if len(eval_savg) >= 5:
    kernel = np.ones(5) / 5
    rolling = np.convolve(eval_savg, kernel, mode='valid')
    rolling_iters = eval_iters[2:2+len(rolling)]
    ax3.plot(rolling_iters, rolling, color='blue', linewidth=2, label='smart_avg (5-eval rolling)')
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(x=exp1_start, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Exp 1 start')

ax3b = ax3.twinx()
ax3b.plot(snap_iters, snap_elos, 'D-', color='orange', markersize=4, alpha=0.6, label='Elo (right axis)')
ax3b.set_ylabel('Elo Rating', color='orange', fontsize=10)
ax3b.tick_params(axis='y', labelcolor='orange')
ax3b.set_ylim(680, 1100)

ax3.set_ylabel('Smart Avg %', fontsize=11)
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_title('Smart Average vs Elo Trajectory', fontsize=12)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 80)

plt.tight_layout()
out = 'data/eval/combined_elo_eval_session35.png'
plt.savefig(out, dpi=150)
print(f'Saved to {out}')

# === ERA ANALYSIS ===
print()
print('=== ERA ANALYSIS (Session 35 Elo measurement) ===')
era_bounds = [
    ('E1 Pre-fix',        0, 280),
    ('E2-3 Type eff',   280, 340),
    ('E4 Stability',    340, 700),
    ('E6 S31 fixes',    724, 940),
    ('E7 S32 disruption', 940, 1500),
    ('E8 S33 stable',  1500, 1785),
    ('E9 Exp1 lam=0.95', 1785, 9999),
]

for era_name, lo, hi in era_bounds:
    era_snaps = [(it, elo) for it, elo in zip(snap_iters, snap_elos) if lo <= it < hi]
    if era_snaps:
        elo_vals = [e for _, e in era_snaps]
        print(f'  {era_name:>22}: n={len(era_snaps):2d}, mean={np.mean(elo_vals):7.1f}, '
              f'min={min(elo_vals):7.0f}, max={max(elo_vals):7.0f}')

print()
base = elos.get('sp1784', 0)
exp1_names = ['sp1789','sp1809','sp1839','sp1879','sp1919','sp1959','sp1984','sp1999']
best_name = max(exp1_names, key=lambda x: elos.get(x, 0))
best_exp1 = elos.get(best_name, 0)
delta = best_exp1 - base

print('=== KEY COMPARISON ===')
print(f'  Baseline (sp1784):      Elo {base:.0f}')
print(f'  Best Exp1 ({best_name}):   Elo {best_exp1:.0f}')
print(f'  Delta:                  +{delta:.0f} Elo')
print(f'  Decision threshold:     +50 Elo')
if delta >= 50:
    print(f'  Verdict:                PASS - hyperparams were the bottleneck')
elif delta >= 30:
    print(f'  Verdict:                MARGINAL (+{delta:.0f}) - positive but below threshold')
else:
    print(f'  Verdict:                Below threshold (+{delta:.0f}) - proceed to Exp 2')

print()
print('=== Exp 1 TRAJECTORY (is it still climbing?) ===')
for name in exp1_names:
    it = int(re.search(r'(\d+)', name).group(1))
    elo = elos.get(name, 0)
    ci = cis.get(name, {})
    lo95 = ci.get('lo95', 0)
    hi95 = ci.get('hi95', 0)
    print(f'  iter {it}: Elo {elo:.0f} [{lo95:.0f}-{hi95:.0f}]  delta vs baseline: +{elo-base:.0f}')
