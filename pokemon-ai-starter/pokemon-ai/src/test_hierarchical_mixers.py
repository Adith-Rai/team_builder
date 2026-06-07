"""Unit tests for SynergisticMixer + TopMixer.

Verifies:
- Distribution honors weights (Monte Carlo)
- Asymmetric rate matches spec (Monte Carlo)
- yield_pair output structure
- Stats accumulate correctly

Run with: python -m pytest test_hierarchical_mixers.py -v
"""
from __future__ import annotations
import random
import pytest

from team_generator import (
    SynergisticMixer,
    TopMixer,
)


class _MockSource:
    """Minimal mock — yield_team returns the source name."""
    def __init__(self, name):
        self.name = name

    def yield_team(self) -> str:
        return f"team_from_{self.name}"


class _MockSourceWithPair(_MockSource):
    """Mock with yield_pair, for testing TopMixer delegation."""
    def yield_pair(self):
        return ("team_from_p1", "team_from_p2", f"{self.name}:p1", f"{self.name}:p2")


# --- SynergisticMixer ---

def test_syn_independent_weight_distribution():
    random.seed(42)
    mixer = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        weights={"hl": 0.6, "gl": 0.4},
        intra_asymmetric_rate=0.0,
    )
    N = 20000
    for _ in range(N):
        mixer.yield_team()
    counts = mixer.selection_stats()["sources"]
    hl_frac = counts["hl"] / sum(counts.values())
    assert 0.58 < hl_frac < 0.62, f"hl fraction off: {hl_frac}"


def test_syn_pair_matched_when_rate_zero():
    random.seed(42)
    mixer = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        weights={"hl": 0.6, "gl": 0.4},
        intra_asymmetric_rate=0.0,
    )
    for _ in range(1000):
        _, _, src1, src2 = mixer.yield_pair()
        assert src1 == src2, f"unexpected mismatch at rate=0: {src1} vs {src2}"
    assert mixer.selection_stats()["pairs"]["asymmetric"] == 0


def test_syn_pair_always_asymmetric_when_rate_one():
    random.seed(42)
    mixer = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        weights={"hl": 0.6, "gl": 0.4},
        intra_asymmetric_rate=1.0,
    )
    for _ in range(500):
        _, _, src1, src2 = mixer.yield_pair()
        assert src1 != src2, f"unexpected match at rate=1: {src1} vs {src2}"
    assert mixer.selection_stats()["pairs"]["matched"] == 0


def test_syn_pair_rate_approximate():
    random.seed(42)
    mixer = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        weights={"hl": 0.6, "gl": 0.4},
        intra_asymmetric_rate=0.3,
    )
    N = 5000
    for _ in range(N):
        mixer.yield_pair()
    pairs = mixer.selection_stats()["pairs"]
    asym_frac = pairs["asymmetric"] / (pairs["matched"] + pairs["asymmetric"])
    assert 0.27 < asym_frac < 0.33, f"asymmetric fraction off: {asym_frac}"


def test_syn_yield_team_uses_independent_sampling():
    # Returns single team (not pair), no asymmetric logic involved
    random.seed(42)
    mixer = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        intra_asymmetric_rate=0.5,
    )
    team = mixer.yield_team()
    assert isinstance(team, str) and team.startswith("team_from_")


# --- TopMixer ---

def test_top_pair_matched_when_top_rate_zero():
    random.seed(42)
    syn = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        weights={"hl": 0.6, "gl": 0.4},
        intra_asymmetric_rate=0.0,
    )
    top = TopMixer(
        procedural=_MockSource("procedural"),
        synergistic=syn,
        syn_pct=0.3,
        top_asymmetric_rate=0.0,
    )
    for _ in range(1000):
        _, _, src1, src2 = top.yield_pair()
        # Both should be procedural OR both should be synergistic
        both_proc = src1 == "procedural" and src2 == "procedural"
        both_syn = src1.startswith("syn:") and src2.startswith("syn:")
        assert both_proc or both_syn, f"unexpected cross-quality: {src1} vs {src2}"


def test_top_pair_always_cross_quality_at_rate_one():
    random.seed(42)
    syn = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        intra_asymmetric_rate=0.0,
    )
    top = TopMixer(
        procedural=_MockSource("procedural"),
        synergistic=syn,
        syn_pct=0.5,
        top_asymmetric_rate=1.0,
    )
    for _ in range(500):
        _, _, src1, src2 = top.yield_pair()
        is_p1_proc = src1 == "procedural"
        is_p2_proc = src2 == "procedural"
        assert is_p1_proc != is_p2_proc, f"unexpected same-quality at rate=1: {src1} vs {src2}"


def test_top_overall_syn_pct_approximate():
    random.seed(42)
    syn = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        weights={"hl": 0.6, "gl": 0.4},
        intra_asymmetric_rate=0.3,
    )
    top = TopMixer(
        procedural=_MockSource("procedural"),
        synergistic=syn,
        syn_pct=0.3,
        top_asymmetric_rate=0.2,
    )
    N = 5000
    proc_yields = 0
    syn_yields = 0
    for _ in range(N):
        team_p1, team_p2, src_p1, src_p2 = top.yield_pair()
        for src in (src_p1, src_p2):
            if src == "procedural":
                proc_yields += 1
            else:
                syn_yields += 1
    total = proc_yields + syn_yields
    syn_frac = syn_yields / total
    # Target per-side syn frequency under defaults:
    # When matched (80%): syn_pct=0.3 → 80%×30%=24% pure-syn
    # When asymmetric (20%): one side is syn → 50%×20%=10%
    # Total per-side syn ≈ 24% + 10% = 34%
    assert 0.30 < syn_frac < 0.38, f"syn fraction off: {syn_frac}"


def test_top_pair_count_buckets():
    random.seed(42)
    syn = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        intra_asymmetric_rate=0.3,
    )
    top = TopMixer(
        procedural=_MockSource("procedural"),
        synergistic=syn,
        syn_pct=0.3,
        top_asymmetric_rate=0.2,
    )
    N = 1000
    for _ in range(N):
        top.yield_pair()
    stats = top.selection_stats()
    pair_total = sum(stats["pairs"].values())
    assert pair_total == N
    asym_frac = stats["pairs"]["asymmetric"] / N
    # top_asymmetric_rate=0.2, so ~20% asymmetric
    assert 0.17 < asym_frac < 0.23, f"top asymmetric frac off: {asym_frac}"


def test_syn_invalid_rate_raises():
    with pytest.raises(ValueError):
        SynergisticMixer(
            sources={"hl": _MockSource("hl")},
            intra_asymmetric_rate=1.5,
        )


def test_top_invalid_pct_raises():
    with pytest.raises(ValueError):
        TopMixer(
            procedural=_MockSource("procedural"),
            synergistic=_MockSource("syn"),
            syn_pct=1.5,
        )


# --- PairedQueueProducer ---

def test_producer_enqueues_correct_count(tmp_path):
    random.seed(42)
    from team_generator import PairedQueueProducer
    syn = SynergisticMixer(
        sources={"hl": _MockSource("hl"), "gl": _MockSource("gl")},
        weights={"hl": 0.6, "gl": 0.4},
        intra_asymmetric_rate=0.3,
    )
    top = TopMixer(
        procedural=_MockSource("procedural"),
        synergistic=syn,
        syn_pct=0.3,
        top_asymmetric_rate=0.2,
    )
    q_p1 = tmp_path / "q_p1"
    q_p2 = tmp_path / "q_p2"
    producer = PairedQueueProducer(top, q_p1, q_p2)
    n = 50
    stats = producer.produce_all(n)
    files_p1 = list(q_p1.glob("*.team"))
    files_p2 = list(q_p2.glob("*.team"))
    assert len(files_p1) == n, f"expected {n} p1 files, got {len(files_p1)}"
    assert len(files_p2) == n, f"expected {n} p2 files, got {len(files_p2)}"
    assert stats["n_pairs"] == n
    assert stats["matched_this_batch"] + stats["asymmetric_this_batch"] == n


def test_producer_pair_fifo_alignment(tmp_path):
    """Verify pair K's two teams sit at the same FIFO position in both queues.

    Critical for matched-pool guarantee: even if pop order differs across
    workers, both queues are populated in lockstep so corresponding files
    line up by FIFO timestamp.
    """
    random.seed(42)
    from team_generator import PairedQueueProducer

    class _IdSource:
        def __init__(self, name, counter):
            self.name = name
            self.counter = counter

        def yield_team(self) -> str:
            v = self.counter[0]
            self.counter[0] += 1
            return f"{self.name}_team_{v}"

    counter_hl = [0]
    counter_gl = [0]
    syn = SynergisticMixer(
        sources={
            "hl": _IdSource("hl", counter_hl),
            "gl": _IdSource("gl", counter_gl),
        },
        intra_asymmetric_rate=0.0,  # always matched within syn
    )
    top = TopMixer(
        procedural=_MockSource("proc"),
        synergistic=syn,
        syn_pct=1.0,  # always syn
        top_asymmetric_rate=0.0,
    )
    q_p1 = tmp_path / "q_p1"
    q_p2 = tmp_path / "q_p2"
    producer = PairedQueueProducer(top, q_p1, q_p2)
    producer.produce_all(20)
    files_p1 = sorted(q_p1.glob("*.team"))
    files_p2 = sorted(q_p2.glob("*.team"))
    # Pair K → file K in each queue (sorted by timestamp). Within each pair,
    # both teams come from the same source (intra=0, syn_pct=1, top=0).
    for f1, f2 in zip(files_p1, files_p2):
        t1 = f1.read_text().strip()
        t2 = f2.read_text().strip()
        # Both should start with same source prefix (hl_ or gl_)
        src1 = t1.split("_")[0]
        src2 = t2.split("_")[0]
        assert src1 == src2, f"pair mismatch in fifo position: {t1} vs {t2}"


def test_producer_zero_pairs(tmp_path):
    from team_generator import PairedQueueProducer
    top = TopMixer(
        procedural=_MockSource("proc"),
        synergistic=_MockSource("syn"),
    )
    producer = PairedQueueProducer(top, tmp_path / "q1", tmp_path / "q2")
    stats = producer.produce_all(0)
    assert stats["n_pairs"] == 0


def test_producer_invalid_n_pairs(tmp_path):
    from team_generator import PairedQueueProducer
    top = TopMixer(
        procedural=_MockSource("proc"),
        synergistic=_MockSource("syn"),
    )
    producer = PairedQueueProducer(top, tmp_path / "q1", tmp_path / "q2")
    with pytest.raises(ValueError):
        producer.produce_all(-1)


# --- StaticTeamPool cache (the perf-critical fix) ---

def test_static_pool_cached_across_builds(tmp_path):
    """build_train_teambuilder should reuse StaticTeamPool across calls
    when paths are the same. Prevents per-iter reload of 180k team files.
    """
    from team_generator import build_train_teambuilder, _STATIC_POOL_CACHE
    # Create a fake teams dir with a few minimal valid Showdown team files
    teams_dir = tmp_path / "fake_teams"
    teams_dir.mkdir()
    team_text = """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
EVs: 252 SpA / 252 Spe / 4 HP
Timid Nature
- Thunderbolt
- Volt Switch
- Surf
- Hidden Power Ice
"""
    (teams_dir / "team_0.gen9ou_team").write_text(team_text)
    (teams_dir / "team_1.gen9ou_team").write_text(team_text)

    # Clear cache to start clean
    _STATIC_POOL_CACHE.clear()

    syn_config = {
        "team_dirs": [(str(teams_dir), 1.0)],
        "team_pct": 0.5,
        "intra_asymmetric_rate": 0.0,
        "top_asymmetric_rate": 0.0,
    }

    # First build → cache miss, instance created
    tb1 = build_train_teambuilder(syn_config=syn_config)
    assert str(teams_dir) in _STATIC_POOL_CACHE, "first build should populate cache"
    cached_instance = _STATIC_POOL_CACHE[str(teams_dir)]

    # Second build → cache hit, same instance reused inside the new mixer
    tb2 = build_train_teambuilder(syn_config=syn_config)
    assert _STATIC_POOL_CACHE[str(teams_dir)] is cached_instance, \
        "second build must reuse cached StaticTeamPool (no reload)"

    # tb1 and tb2 themselves are DIFFERENT mixer instances (so selection_stats
    # starts fresh each iter), but they share the underlying source pool.
    assert tb1 is not tb2, "mixer wrapping should rebuild fresh (per-iter stats)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
