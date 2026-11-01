"""PID attribution must include CLI descendants, excluding unrelated GPU jobs."""


def test_samples_attribute_gpu_and_rss_to_task_session():
    from scripts.ec2.sample_resources import parse_sample

    rows = parse_sample(
        "100 100 150\n101 100 12000\n102 100 600\n999 999 777\n",
        "101, 42\n999, 5000\n",
        100,
    )
    assert rows == [(100, 150, 0), (101, 12000, 42), (102, 600, 0)]


def test_samples_reject_missing_gpu_measurement():
    import pytest

    from scripts.ec2.sample_resources import parse_sample

    with pytest.raises(ValueError):
        parse_sample("101 100 12000", "101, [N/A]", 100)
