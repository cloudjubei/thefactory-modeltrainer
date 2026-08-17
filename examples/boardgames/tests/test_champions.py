from harness.champions import (
    best_champion_path,
    champion_generation,
    champion_is_distilled,
    champion_pool_paths,
    promote_champion,
)


def _saver(text):
    return lambda path: open(path, "w").write(text)


def test_distilled_flag_drives_the_warm_start_decision(tmp_path):
    # A fresh store / a non-distilled champion → not distilled (a distillation run must start FRESH from it).
    assert champion_is_distilled("connect4", store_dir=tmp_path) is False
    promote_champion(_saver("w1"), "connect4", store_dir=tmp_path)
    assert champion_is_distilled("connect4", store_dir=tmp_path) is False
    # Once a DISTILLED champion is crowned, warm-starting from it is safe.
    promote_champion(_saver("w2"), "connect4", store_dir=tmp_path, distilled=True)
    assert champion_is_distilled("connect4", store_dir=tmp_path) is True


def test_empty_store_has_no_champion(tmp_path):
    assert champion_generation("connect4", store_dir=tmp_path) == 0
    assert best_champion_path("connect4", store_dir=tmp_path) is None
    assert champion_pool_paths("connect4", store_dir=tmp_path) == []


def test_promotion_increments_generation_and_sets_best(tmp_path):
    p1, g1 = promote_champion(_saver("w1"), "connect4", store_dir=tmp_path)
    assert g1 == 1
    assert best_champion_path("connect4", store_dir=tmp_path) == p1
    p2, g2 = promote_champion(_saver("w2"), "connect4", store_dir=tmp_path)
    assert g2 == 2
    assert best_champion_path("connect4", store_dir=tmp_path) == p2  # newest is best
    assert champion_generation("connect4", store_dir=tmp_path) == 2


def test_pool_returns_recent_champions_newest_last(tmp_path):
    paths = [promote_champion(_saver(f"w{i}"), "connect4", store_dir=tmp_path)[0] for i in range(4)]
    pool = champion_pool_paths("connect4", k=2, store_dir=tmp_path)
    assert pool == paths[-2:]


def test_missing_weights_file_is_ignored(tmp_path):
    p1, _ = promote_champion(_saver("w1"), "connect4", store_dir=tmp_path)
    __import__("os").remove(p1)
    assert best_champion_path("connect4", store_dir=tmp_path) is None
    assert champion_pool_paths("connect4", store_dir=tmp_path) == []
