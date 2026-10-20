from datetime import date, datetime

import polars as pl
import pytest

from vvs import schemas


def test_hypothesis_id_sorted_and_without_reserved():
    assert schemas.hypothesis_id(["b", "a", "undecided", "a"]) == "a+b"


def test_validate_polls_rejects_reversed_fieldwork():
    df = pl.DataFrame(
        {
            "poll_id": ["x"], "election": [2022], "pollster": ["ifop"], "sponsor": [None],
            "fieldwork_start": [date(2022, 4, 2)], "fieldwork_end": [date(2022, 4, 1)],
            "publish_date": [None], "sample_size": [1000], "mode": ["online"],
            "source": ["wikipedia"], "source_url": ["u"], "ingested_at": [datetime(2026, 1, 1)],
        }
    )
    with pytest.raises(ValueError):
        schemas.validate_polls(df)
