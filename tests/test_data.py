import numpy as np

from tinysleep.data import EPOCH_N, epoch_signal, fold_split, labels_from_annotations, trim_wake


def test_labels_and_merge_n3_n4():
    on = [0, 60, 120, 150]
    dur = [60, 60, 30, 30]
    desc = ["Sleep stage W", "Sleep stage 3", "Sleep stage 4", "Movement time"]
    y = labels_from_annotations(on, dur, desc, 7)
    assert y.tolist() == [0, 0, 3, 3, 3, -1, -1]


def test_trim_wake_keeps_30_min():
    y = np.array([0] * 200 + [2] * 10 + [0] * 200)
    sl = trim_wake(y)
    assert (sl.start, sl.stop) == (140, 270)


def test_epoch_signal_drops_unscored():
    y = np.array([0, 1, -1, 2])
    X, yy = epoch_signal(np.arange(4 * EPOCH_N, dtype=float), y)
    assert yy.tolist() == [0, 1, 2] and X.shape == (3, EPOCH_N)
    assert X[2, 0] == 3 * EPOCH_N  # third kept epoch is the fourth original one


def test_subject_wise_split_has_no_leakage():
    seen_test = []
    for f in range(5):
        tr, va, te = fold_split(f)
        assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))
        assert len(tr) + len(va) + len(te) == 20
        seen_test += te
    assert sorted(seen_test) == list(range(20))  # every subject tested exactly once
