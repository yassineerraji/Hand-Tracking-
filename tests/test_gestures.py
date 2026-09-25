from types import SimpleNamespace

from hand_tracking.gestures import count_raised_fingers, viewer_handedness


def make_hand(thumb_out: bool, raised: set[int]):
    """Builds 21 fake landmarks for an upright hand (smaller y = higher)."""
    lm = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
    lm[17] = SimpleNamespace(x=0.6, y=0.7)          # pinky-side of palm
    lm[2] = SimpleNamespace(x=0.45, y=0.7)          # thumb base joint
    lm[4] = SimpleNamespace(x=0.3 if thumb_out else 0.55, y=0.65)
    for tip in (8, 12, 16, 20):
        lm[tip - 2] = SimpleNamespace(x=0.5, y=0.5)  # pip joint
        lm[tip] = SimpleNamespace(x=0.5, y=0.3 if tip in raised else 0.6)
    return lm


def test_closed_fist_counts_zero():
    assert count_raised_fingers(make_hand(False, set())) == 0


def test_open_hand_counts_five():
    assert count_raised_fingers(make_hand(True, {8, 12, 16, 20})) == 5


def test_peace_sign_counts_two():
    assert count_raised_fingers(make_hand(False, {8, 12})) == 2


def test_handedness_is_mirrored():
    assert viewer_handedness("Left") == "Right"
    assert viewer_handedness("Right") == "Left"
