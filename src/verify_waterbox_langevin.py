#!/usr/bin/env python
"""Synthetic test of waterbox_langevin.py's pure-logic pieces (Snapshot,
ReplicaState) in isolation from any real dynamics - the same "verify before
trusting" discipline as verify_boltzmann_estimator.py, scoped to what's
actually testable without ase/torchmdnet on this checkout (see
waterbox_langevin.py's module docstring for the testable/not-testable
split). run_stable_sampling_phase itself (the ASE-driven half) is NOT
covered here - see waterbox_langevin.py's --smoke-test for that, which
needs the training box.

Usage (runs anywhere, no ase/torchmdnet needed):
    python src/verify_waterbox_langevin.py
"""

from __future__ import annotations

import numpy as np

from waterbox_langevin import ReplicaState, Snapshot


def _snap(step, value):
    """A tiny synthetic snapshot whose positions are filled with `value` -
    lets a test identify which snapshot survived a sequence of operations
    just by inspecting its contents, without needing real atomic
    coordinates."""
    return Snapshot(
        positions=np.full((3, 3), float(value)),
        velocities=np.zeros((3, 3)),
        step=step,
    )


def test_initial_state_is_last_good():
    print("=== test 1: initial snapshot is last_good before anything happens ===")
    init = _snap(0, 0)
    state = ReplicaState(init)
    assert state.last_good is init
    assert state.n_rewinds == 0
    assert state.collected == []
    print("  PASS\n")


def test_record_stable_frame_updates_last_good_and_collected():
    print("=== test 2: record_stable_frame appends + updates last_good ===")
    init = _snap(0, 0)
    state = ReplicaState(init)
    s1, s2 = _snap(1, 1), _snap(2, 2)
    state.record_stable_frame(s1)
    state.record_stable_frame(s2)
    assert state.collected == [s1, s2]
    assert state.last_good is s2
    print("  PASS\n")


def test_rewind_before_any_stable_frame_returns_initial():
    print("=== test 3: rewind before any stable frame returns the initial snapshot ===")
    init = _snap(0, 0)
    state = ReplicaState(init)
    returned = state.rewind()
    assert returned is init
    assert state.n_rewinds == 1
    assert state.collected == []
    print("  PASS\n")


def test_rewind_discards_current_window_but_keeps_last_good():
    print("=== test 4: rewind discards the invalidated window, keeps the last GOOD snapshot ===")
    init = _snap(0, 0)
    state = ReplicaState(init)
    s1 = _snap(1, 1)
    state.record_stable_frame(s1)  # last_good is now s1
    s2 = _snap(2, 2)
    state.record_stable_frame(s2)  # collected = [s1, s2], last_good = s2

    # Simulate: after s2, the NEXT step was unstable - the window ending in
    # that unstable step is discarded, but last_good (s2, the most recent
    # frame that DID pass) is what gets reloaded.
    returned = state.rewind()
    assert returned is s2, "rewind should return the most recent STABLE snapshot, not the initial one"
    assert state.collected == [], "rewind must clear the now-invalidated window"
    assert state.n_rewinds == 1
    print("  PASS\n")


def test_take_learn_window_subsamples_and_clears():
    print("=== test 5: take_learn_window subsamples every stride-th frame, then clears ===")
    init = _snap(0, 0)
    state = ReplicaState(init)
    frames = [_snap(i, i) for i in range(1, 11)]  # 10 frames
    for f in frames:
        state.record_stable_frame(f)

    window = state.take_learn_window(stride=3)
    expected = frames[::3]  # frames[0], frames[3], frames[6], frames[9]
    assert window == expected, f"expected {[s.step for s in expected]}, got {[s.step for s in window]}"
    assert state.collected == [], "take_learn_window must clear the buffer for the next phase"
    print(f"  10 collected frames, stride=3 -> steps {[s.step for s in window]}")
    print("  PASS\n")


def test_take_learn_window_empty_and_sparse_cases():
    print("=== test 6: take_learn_window edge cases (empty buffer, stride > n collected) ===")
    init = _snap(0, 0)
    state = ReplicaState(init)

    empty_window = state.take_learn_window(stride=5)
    assert empty_window == [], "no collected frames -> empty window, not an error (caller must handle this)"
    print("  empty buffer -> [] (as expected; boltzmann_estimator_pseudo_loss will reject N<2 downstream)")

    frames = [_snap(i, i) for i in range(1, 4)]  # only 3 frames
    for f in frames:
        state.record_stable_frame(f)
    sparse_window = state.take_learn_window(stride=10)
    assert sparse_window == [frames[0]], "stride > n_collected should still return the first frame (Python slice semantics)"
    print(f"  3 collected frames, stride=10 -> steps {[s.step for s in sparse_window]}")

    try:
        state.take_learn_window(stride=0)
        raise AssertionError("expected ValueError for stride=0")
    except ValueError as e:
        print(f"  stride=0 correctly raised: {e}")
    print("  PASS\n")


def test_full_simulate_rewind_collect_sequence():
    print("=== test 7: a realistic simulate/rewind/collect sequence end-to-end ===")
    init = _snap(0, 0)
    state = ReplicaState(init)

    # Phase A: 4 stable steps.
    for i in range(1, 5):
        state.record_stable_frame(_snap(i, i))
    assert [s.step for s in state.collected] == [1, 2, 3, 4]

    # Step 5 is unstable -> rewind. Should revert to step 4 (last good),
    # discard the phase-A collection.
    reloaded = state.rewind()
    assert reloaded.step == 4
    assert state.collected == []
    assert state.n_rewinds == 1

    # Phase B (post-rewind): 6 more stable steps, numbered continuing from
    # the reload point for realism (a real caller would re-draw velocities
    # here but keep stepping the simulation step counter forward).
    for i in range(5, 11):
        state.record_stable_frame(_snap(i, i))
    window = state.take_learn_window(stride=2)
    assert [s.step for s in window] == [5, 7, 9]
    assert state.collected == []
    print(f"  post-rewind window (stride=2): steps {[s.step for s in window]}, n_rewinds={state.n_rewinds}")
    print("  PASS\n")


if __name__ == "__main__":
    test_initial_state_is_last_good()
    test_record_stable_frame_updates_last_good_and_collected()
    test_rewind_before_any_stable_frame_returns_initial()
    test_rewind_discards_current_window_but_keeps_last_good()
    test_take_learn_window_subsamples_and_clears()
    test_take_learn_window_empty_and_sparse_cases()
    test_full_simulate_rewind_collect_sequence()
    print("All waterbox_langevin.py (pure-logic) checks passed.")
