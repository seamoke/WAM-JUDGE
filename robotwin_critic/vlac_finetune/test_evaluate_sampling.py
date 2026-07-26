from .evaluate_vlac import select_rows


def make_rows(task: str, label: int, count: int):
    return [
        {"task": task, "label": label, "index": index}
        for index in range(count)
    ]


def test_stratified_sampling_is_deterministic_and_balanced():
    rows = (
        make_rows("task-a", -1, 20)
        + make_rows("task-a", 1, 20)
        + make_rows("task-b", -1, 20)
        + make_rows("task-b", 1, 20)
    )
    key_fn = lambda row: (row["task"], row["label"])

    first = select_rows(rows, 12, "stratified", 7, key_fn=key_fn)
    second = select_rows(rows, 12, "stratified", 7, key_fn=key_fn)

    assert first == second
    assert {
        key: sum(key_fn(row) == key for row in first)
        for key in {key_fn(row) for row in rows}
    } == {
        ("task-a", -1): 3,
        ("task-a", 1): 3,
        ("task-b", -1): 3,
        ("task-b", 1): 3,
    }


def test_first_sampling_preserves_existing_behavior():
    rows = make_rows("task-a", 1, 8)

    assert select_rows(
        rows,
        3,
        "first",
        42,
        key_fn=lambda row: row["task"],
    ) == rows[:3]
