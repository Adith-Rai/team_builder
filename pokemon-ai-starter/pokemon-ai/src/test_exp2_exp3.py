"""Tests for Exp 2 (slot permutation) and Exp 3 (PFSP opponent sampling).

Run: python test_exp2_exp3.py
"""
import sys, os, json, random, copy
from collections import Counter
from pathlib import Path

# ── Slot Permutation Tests ──────────────────────────────────────────────────

def make_fake_pokemon(species_id, move_ids=(10, 20, 30, 40)):
    """Create a minimal pokemon feature dict matching _encode_pokemon output."""
    # Continuous: 285 dims total, last 92 = 4 moves * 23 each
    base_cont = [float(species_id)] * (285 - 92)
    # Each move's 23 features: [move_id * 100 + i for i in range(23)]
    move_cont = []
    for mi, mid in enumerate(move_ids):
        move_cont.extend([float(mid * 100 + j) for j in range(23)])
    cont = base_cont + move_cont

    return {
        "ids": {
            "species": species_id, "item": species_id + 100, "ability": species_id + 200,
            "move0": move_ids[0], "move1": move_ids[1], "move2": move_ids[2], "move3": move_ids[3],
        },
        "banks": {
            "hp_pct": species_id, "level": 50, "weight": 10, "height": 5,
            "stat_hp": 100, "stat_atk": 100, "stat_def": 100,
            "stat_spa": 100, "stat_spd": 100, "stat_spe": 100,
        },
        "continuous": cont,
    }


def test_active_never_shuffled():
    """Active pokemon (slot 0) must never move to another position."""
    from features import _permute_team
    team = [make_fake_pokemon(i, (i*10+1, i*10+2, i*10+3, i*10+4)) for i in range(6)]
    active_species = team[0]["ids"]["species"]

    for _ in range(500):
        perm = _permute_team(team)
        assert perm[0]["ids"]["species"] == active_species, \
            f"Active pokemon moved! Got species {perm[0]['ids']['species']}, expected {active_species}"
    print("  PASS: active pokemon never shuffled (500 trials)")


def test_bench_all_orderings():
    """All 120 bench orderings (5!) should appear with roughly equal frequency."""
    from features import _permute_team
    team = [make_fake_pokemon(i) for i in range(6)]
    ordering_counts = Counter()

    n_trials = 12000  # expect ~100 per ordering
    for _ in range(n_trials):
        perm = _permute_team(team)
        bench_order = tuple(p["ids"]["species"] for p in perm[1:])
        ordering_counts[bench_order] += 1

    n_orderings = len(ordering_counts)
    assert n_orderings == 120, f"Expected 120 bench orderings, got {n_orderings}"
    expected = n_trials / 120
    for order, count in ordering_counts.items():
        assert count > expected * 0.5, f"Ordering {order} underrepresented: {count} (expected ~{expected:.0f})"
        assert count < expected * 1.8, f"Ordering {order} overrepresented: {count} (expected ~{expected:.0f})"
    print(f"  PASS: all 120 bench orderings appear (min={min(ordering_counts.values())}, max={max(ordering_counts.values())})")


def test_move_permutation():
    """Move features within each pokemon should be randomly permuted."""
    from features import _permute_team
    team = [make_fake_pokemon(0, (10, 20, 30, 40))]  # just 1 pokemon to simplify
    # Pad to 6 for _permute_team
    for i in range(5):
        team.append(make_fake_pokemon(i + 1))

    move_orderings = Counter()
    for _ in range(2400):  # expect ~100 per ordering (24 orderings for 4 moves)
        perm = _permute_team(team)
        p = perm[0]  # active pokemon
        move_order = tuple(p["ids"][f"move{i}"] for i in range(4))
        move_orderings[move_order] += 1

    n_orderings = len(move_orderings)
    assert n_orderings == 24, f"Expected 24 move orderings, got {n_orderings}"
    print(f"  PASS: all 24 move orderings appear (min={min(move_orderings.values())}, max={max(move_orderings.values())})")


def test_move_cont_matches_ids():
    """After permutation, move continuous features must match permuted move IDs."""
    from features import _permute_team, MOVE_CONT_PER_SLOT, N_MOVE_SLOTS

    team = [make_fake_pokemon(0, (10, 20, 30, 40))] + [make_fake_pokemon(i+1) for i in range(5)]
    for _ in range(200):
        perm = _permute_team(team)
        p = perm[0]
        cont = p["continuous"]
        base = len(cont) - N_MOVE_SLOTS * MOVE_CONT_PER_SLOT
        for mi in range(N_MOVE_SLOTS):
            mid = p["ids"][f"move{mi}"]
            # Our fake move cont starts with mid * 100 + 0
            expected_first = float(mid * 100)
            actual_first = cont[base + mi * MOVE_CONT_PER_SLOT]
            assert actual_first == expected_first, \
                f"Move {mi}: id={mid}, expected cont start={expected_first}, got {actual_first}"
    print("  PASS: move continuous features match permuted move IDs (200 trials)")


def test_no_mutation_of_input():
    """_permute_team must not mutate the original team list or its dicts."""
    from features import _permute_team
    team = [make_fake_pokemon(i) for i in range(6)]
    original = copy.deepcopy(team)

    for _ in range(50):
        _permute_team(team)

    for i in range(6):
        assert team[i]["ids"]["species"] == original[i]["ids"]["species"], \
            f"Input mutated! Pokemon {i} species changed"
        assert team[i]["continuous"] == original[i]["continuous"], \
            f"Input mutated! Pokemon {i} continuous changed"
        for mi in range(4):
            assert team[i]["ids"][f"move{mi}"] == original[i]["ids"][f"move{mi}"], \
                f"Input mutated! Pokemon {i} move{mi} changed"
    print("  PASS: input not mutated (50 trials)")


def test_training_false_no_augmentation():
    """build_turn_batch with training=False must produce identical output every time."""
    from features import build_turn_batch
    import torch

    team = [make_fake_pokemon(i, (i*10+1, i*10+2, i*10+3, i*10+4)) for i in range(6)]
    import numpy as np
    feat = {
        "our_pokemon": team,
        "opp_pokemon": [make_fake_pokemon(i + 10) for i in range(6)],
        "field": {"continuous": [0.0] * 52, "banks": {"turn": 1, "weather_dur": 0, "terrain_dur": 0, "tr_dur": 0}},
        "transition": {"continuous": [0.0] * 51, "ids": {"our_action": 0, "opp_action": 0}},
        "legal_mask": np.array([1,1,1,1,1,1,0,0,0], dtype=np.float32),
        "active_moves": [
            {"move_id": 1, "bp_int": 10, "acc_int": 100, "pp_int": 15, "priority_int": 6, "continuous": [0.0]*109},
            None, None, None,
        ],
        "switch_slots": [
            {"species_id": 2, "continuous": [0.0]*30},
            {"species_id": 3, "continuous": [0.0]*30},
            None, None, None,
        ],
    }

    b1 = build_turn_batch(feat, training=False)
    b2 = build_turn_batch(feat, training=False)

    for key in b1:
        if isinstance(b1[key], torch.Tensor):
            assert torch.equal(b1[key], b2[key]), f"training=False not deterministic for key={key}"
        elif isinstance(b1[key], dict):
            for k2 in b1[key]:
                if isinstance(b1[key][k2], torch.Tensor):
                    assert torch.equal(b1[key][k2], b2[key][k2]), f"training=False not deterministic for key={key}.{k2}"
    print("  PASS: training=False produces identical output")


def test_tensor_shapes_unchanged():
    """Tensor shapes must be the same with and without augmentation."""
    from features import build_turn_batch
    import numpy as np

    team = [make_fake_pokemon(i) for i in range(6)]
    feat = {
        "our_pokemon": team,
        "opp_pokemon": [make_fake_pokemon(i + 10) for i in range(6)],
        "field": {"continuous": [0.0] * 52, "banks": {"turn": 1, "weather_dur": 0, "terrain_dur": 0, "tr_dur": 0}},
        "transition": {"continuous": [0.0] * 51, "ids": {"our_action": 0, "opp_action": 0}},
        "legal_mask": np.array([1,1,1,1,1,1,0,0,0], dtype=np.float32),
        "active_moves": [
            {"move_id": 1, "bp_int": 10, "acc_int": 100, "pp_int": 15, "priority_int": 6, "continuous": [0.0]*109},
            None, None, None,
        ],
        "switch_slots": [
            {"species_id": 2, "continuous": [0.0]*30},
            None, None, None, None,
        ],
    }

    b_no_aug = build_turn_batch(feat, training=False)
    b_aug = build_turn_batch(feat, training=True)

    for key in b_no_aug:
        if isinstance(b_no_aug[key], dict):
            for k2 in b_no_aug[key]:
                v1, v2 = b_no_aug[key][k2], b_aug[key][k2]
                if hasattr(v1, 'shape'):
                    assert v1.shape == v2.shape, f"Shape mismatch for {key}.{k2}: {v1.shape} vs {v2.shape}"
        elif hasattr(b_no_aug[key], 'shape'):
            assert b_no_aug[key].shape == b_aug[key].shape, \
                f"Shape mismatch for {key}: {b_no_aug[key].shape} vs {b_aug[key].shape}"
    print("  PASS: tensor shapes unchanged with augmentation")


def test_switch_slots_unaffected():
    """switch_slots and legal_mask must NOT change with augmentation."""
    from features import build_turn_batch
    import torch, numpy as np

    team = [make_fake_pokemon(i) for i in range(6)]
    feat = {
        "our_pokemon": team,
        "opp_pokemon": [make_fake_pokemon(i + 10) for i in range(6)],
        "field": {"continuous": [0.0] * 52, "banks": {"turn": 1, "weather_dur": 0, "terrain_dur": 0, "tr_dur": 0}},
        "transition": {"continuous": [0.0] * 51, "ids": {"our_action": 0, "opp_action": 0}},
        "legal_mask": np.array([1,0,1,0,1,0,1,0,1], dtype=np.float32),
        "active_moves": [
            {"move_id": 5, "bp_int": 80, "acc_int": 100, "pp_int": 10, "priority_int": 6, "continuous": [1.0]*109},
            None, None, None,
        ],
        "switch_slots": [
            {"species_id": 99, "continuous": [1.0]*30},
            {"species_id": 88, "continuous": [2.0]*30},
            None, None, None,
        ],
    }

    b_base = build_turn_batch(feat, training=False)
    for _ in range(50):
        b_aug = build_turn_batch(feat, training=True)
        assert torch.equal(b_base["switch_ids"], b_aug["switch_ids"]), "switch_ids changed!"
        assert torch.equal(b_base["switch_cont"], b_aug["switch_cont"]), "switch_cont changed!"
        assert torch.equal(b_base["legal_mask"], b_aug["legal_mask"]), "legal_mask changed!"
        assert torch.equal(b_base["active_move_ids"], b_aug["active_move_ids"]), "active_move_ids changed!"
    print("  PASS: switch_slots, active_moves, legal_mask unaffected by augmentation (50 trials)")


# ── PFSP Tests ──────────────────────────────────────────────────────────────

def test_pfsp_all_05_is_uniform():
    """With all win rates at 0.5, PFSP should sample approximately uniformly."""
    from rl_collection import pfsp_sample
    pool = [f"snap_{i}.pt" for i in range(100)]
    win_rates = {}  # all default to 0.5

    counts = Counter()
    n_trials = 5000
    for _ in range(n_trials):
        selected = pfsp_sample(pool, win_rates, n_opponents=15)
        for s in selected:
            counts[s] += 1

    # Each checkpoint should appear ~750 times (5000 * 15 / 100)
    expected = n_trials * 15 / 100
    vals = list(counts.values())
    assert len(counts) == 100, f"Not all checkpoints sampled! Got {len(counts)}"
    cv = (max(vals) - min(vals)) / expected  # coefficient of variation
    assert cv < 1.0, f"Too much variance for uniform: cv={cv:.2f}, min={min(vals)}, max={max(vals)}"
    print(f"  PASS: all-0.5 gives near-uniform (min={min(vals)}, max={max(vals)}, expected~{expected:.0f})")


def test_pfsp_weight_calculation():
    """Hard opponents (low win rate) should be sampled much more than easy ones."""
    from rl_collection import pfsp_sample
    pool = [f"snap_{i}.pt" for i in range(50)]
    # Make first 10 very hard (win rate 0.1), rest very easy (win rate 0.9)
    win_rates = {}
    for i in range(10):
        win_rates[pool[i]] = [1, 10]  # 10% win rate
    for i in range(10, 50):
        win_rates[pool[i]] = [9, 10]  # 90% win rate

    hard_count, easy_count = 0, 0
    n_trials = 3000
    for _ in range(n_trials):
        selected = pfsp_sample(pool, win_rates, n_opponents=15, uniform_frac=0.0)
        for s in selected:
            idx = int(s.split("_")[1].split(".")[0])
            if idx < 10:
                hard_count += 1
            else:
                easy_count += 1

    # With 10 hard opponents: they should almost always ALL be selected (10/15 slots).
    # Remaining 5 slots go to easy opponents. So ratio should be ~2.0 (10:5).
    # The key metric: per-checkpoint selection rate. Hard checkpoints should be picked ~10x more often.
    hard_per_ckpt = hard_count / 10
    easy_per_ckpt = easy_count / 40
    ckpt_ratio = hard_per_ckpt / max(1, easy_per_ckpt)
    assert ckpt_ratio > 5.0, f"PFSP not prioritizing hard opponents! per-ckpt ratio={ckpt_ratio:.1f}"
    print(f"  PASS: hard opponents heavily prioritized (per-ckpt ratio={ckpt_ratio:.1f}x, hard_rate={hard_per_ckpt:.0f}, easy_rate={easy_per_ckpt:.0f})")


def test_pfsp_latest_always_included():
    """The latest snapshot must always be in the selection."""
    from rl_collection import pfsp_sample
    pool = [f"snap_{i}.pt" for i in range(100)]
    latest = "snap_99.pt"
    # Make latest very easy so PFSP would normally skip it
    win_rates = {latest: [99, 100]}  # 99% win rate

    for _ in range(200):
        selected = pfsp_sample(pool, win_rates, n_opponents=15, latest_snapshot=latest)
        assert latest in selected, f"Latest snapshot not in selection!"
    print("  PASS: latest snapshot always included (200 trials)")


def test_pfsp_no_duplicates():
    """Selection should never contain duplicate checkpoints."""
    from rl_collection import pfsp_sample
    pool = [f"snap_{i}.pt" for i in range(100)]
    win_rates = {pool[0]: [1, 100]}  # one very hard opponent

    for _ in range(200):
        selected = pfsp_sample(pool, win_rates, n_opponents=15, latest_snapshot=pool[-1])
        assert len(selected) == len(set(selected)), f"Duplicates in selection: {selected}"
    print("  PASS: no duplicates (200 trials)")


def test_pfsp_small_pool():
    """When pool <= n_opponents, return entire pool."""
    from rl_collection import pfsp_sample
    pool = [f"snap_{i}.pt" for i in range(10)]
    win_rates = {}
    selected = pfsp_sample(pool, win_rates, n_opponents=15)
    assert set(selected) == set(pool), f"Small pool not fully returned"
    print("  PASS: small pool returns all")


def test_win_rates_json_roundtrip():
    """Win rates should survive JSON save/load."""
    import tempfile
    wr = {"snap_0.pt": [10, 20], "snap_1.pt": [5, 15]}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(wr, f)
        tmp = f.name
    try:
        with open(tmp) as f:
            loaded = json.load(f)
        assert loaded == wr, f"JSON round-trip failed: {loaded} != {wr}"
        print("  PASS: win_rates JSON round-trip")
    finally:
        os.unlink(tmp)


# ── Run all tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    print("=" * 60)
    print("SLOT PERMUTATION TESTS")
    print("=" * 60)
    test_active_never_shuffled()
    test_bench_all_orderings()
    test_move_permutation()
    test_move_cont_matches_ids()
    test_no_mutation_of_input()
    test_training_false_no_augmentation()
    test_tensor_shapes_unchanged()
    test_switch_slots_unaffected()

    print()
    print("=" * 60)
    print("PFSP TESTS")
    print("=" * 60)
    test_pfsp_all_05_is_uniform()
    test_pfsp_weight_calculation()
    test_pfsp_latest_always_included()
    test_pfsp_no_duplicates()
    test_pfsp_small_pool()
    test_win_rates_json_roundtrip()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
