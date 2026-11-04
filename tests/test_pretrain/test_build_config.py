"""The exact reviewed include/exclude lists and strict offline recipe schema."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from demand_forecast.pretrain.build_config import CorpusBuildConfig
from demand_forecast.pretrain.corpus import CHRONOS_REVISION

RECIPE = Path(__file__).resolve().parents[2] / "configs/corpus-v2.yaml"


def test_reviewed_recipe_is_exhaustive_and_pinned():
    cfg = CorpusBuildConfig.from_yaml(RECIPE)
    chronos, gift = cfg.sources
    assert cfg.version == 2 and cfg.commercial_use is False
    assert (chronos.repo, chronos.revision, chronos.kind) == (
        "autogluon/chronos_datasets",
        "eeecad0b82a8c237e212ce6f8d1abecb513e2cec",
        "chronos_real",
    )
    assert chronos.revision == CHRONOS_REVISION
    assert (gift.repo, gift.revision, gift.kind) == (
        "Salesforce/GiftEvalPretrain",
        "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874",
        "gifteval_pretrain",
    )
    assert gift.bundle_terms == "research-only (dataset card)"
    assert [e.config for e in chronos.include] == [
        "electricity_15min",
        "exchange_rate",
        "monash_australian_electricity",
        "monash_car_parts",
        "monash_covid_deaths",
        "monash_electricity_hourly",
        "monash_electricity_weekly",
        "monash_london_smart_meters",
        "monash_saugeenday",
        "monash_temperature_rain",
        "monash_traffic",
        "monash_weather",
        "wiki_daily_100k",
        "wind_farms_daily",
        "wind_farms_hourly",
    ]
    assert {
        bucket: [e.folder for e in gift.include if e.bucket == bucket]
        for bucket in ("yes", "nc", "sa")
    } == {
        "yes": [
            "beijing_air_quality",
            "cdc_fluview_ilinet",
            "cdc_fluview_who_nrevss",
            "ideal",
            "lcl",
            "project_tycho",
            "sceaux",
            "solar_power",
            "wiki-rolling_nips",
            "wind_power",
        ],
        "nc": ["residential_load_power", "residential_pv_power", "sunspot_with_missing"],
        "sa": ["bdg-2_bear", "bdg-2_fox", "bdg-2_panther", "bdg-2_rat"],
    }
    assert chronos.excluded.model_dump(exclude_defaults=True) == {
        "fev_overlap": ["m5", "ercot", "solar", "solar_1h", "monash_hospital", "monash_fred_md"],
        "external_validation": ["monash_nn5_weekly"],
        "unlicensed_or_unverified": [
            "dominick",
            "m4_daily",
            "m4_hourly",
            "m4_monthly",
            "m4_quarterly",
            "m4_weekly",
            "m4_yearly",
            "mexico_city_bikes",
            "nn5",
            "uber_tlc_daily",
            "uber_tlc_hourly",
            "ushcn_daily",
            "taxi_1h",
            "taxi_30min",
            "monash_kdd_cup_2018",
        ],
        "unclear_competition_grant": [
            "monash_m1_monthly",
            "monash_m1_quarterly",
            "monash_m1_yearly",
            "monash_m3_monthly",
            "monash_m3_quarterly",
            "monash_m3_yearly",
            "monash_tourism_monthly",
            "monash_tourism_quarterly",
            "monash_tourism_yearly",
            "monash_cif_2016",
            "monash_rideshare",
            "monash_pedestrian_counts",
        ],
        "ruled_out": ["weatherbench_*"],
        "existing_corpus": ["training_corpus_tsmixup_10m", "training_corpus_kernel_synth_1m"],
    }
    assert gift.excluded.model_dump(exclude_defaults=True) == {
        "fev_overlap": [
            "m5",
            "favorita_sales",
            "favorita_transactions",
            "kdd2022",
            "fred_md",
            "bull",
            "cockatoo",
            "hog",
            "pdb",
            "gfc12_load",
            "gfc14_load",
            "gfc17_load",
        ],
        "duplicate_of_chronos": [
            "australian_electricity_demand",
            "london_smart_meters_with_missing",
            "traffic_hourly",
            "traffic_weekly",
            "weather",
            "wind_farms_with_missing",
        ],
        "unclear": [
            "bitcoin_with_missing",
            "borealis",
            "china_air_quality",
            "cif_2016_12",
            "cif_2016_6",
            "covid19_energy",
            "covid_mobility",
            "elecdemand",
            "elf",
            "extended_web_traffic_with_missing",
            "kaggle_web_traffic_weekly",
            "m1_monthly",
            "m1_quarterly",
            "m1_yearly",
            "monash_m3_monthly",
            "monash_m3_other",
            "monash_m3_quarterly",
            "monash_m3_yearly",
            "nn5_daily_with_missing",
            "nn5_weekly",
            "oikolab_weather",
            "pedestrian_counts",
            "rideshare_with_missing",
            "smart",
            "spain",
            "taxi_30min",
            "tourism_monthly",
            "tourism_quarterly",
            "tourism_yearly",
            "uber_tlc_daily",
            "uber_tlc_hourly",
            "vehicle_trips_with_missing",
        ],
        "not_relevant": "every folder not named above (climate simulations, cloud traces, PEMS/metro traffic, buildings_900k, subseasonal*, godaddy, and others)",
    }
    special = next(e for e in chronos.include if e.config == "monash_temperature_rain")
    assert (special.licence, special.card_licence, special.bucket) == (
        "CC BY-SA 3.0 AU",
        "CC BY 4.0",
        "sa",
    )
    assert all(e.licence and e.attribution and e.url for s in cfg.sources for e in s.include)
    assert cfg.filters.model_dump() == dict(
        min_points=64,
        drop_constant=True,
        drop_all_nan=True,
        dedupe="value_hash",
        max_series_per_folder=None,
    )
    assert cfg.shards.model_dump() == dict(max_series=20000, max_bytes=268435456)


@pytest.mark.parametrize("section", ["root", "source", "include", "excluded", "filters", "shards"])
def test_recipe_forbids_unknown_keys(section):
    raw = CorpusBuildConfig.from_yaml(RECIPE).model_dump()
    target = {
        "root": raw,
        "source": raw["sources"][0],
        "include": raw["sources"][0]["include"][0],
        "excluded": raw["sources"][0]["excluded"],
        "filters": raw["filters"],
        "shards": raw["shards"],
    }[section]
    target["typo"] = 1
    with pytest.raises(ValidationError, match="typo"):
        CorpusBuildConfig.model_validate(raw)


@pytest.mark.parametrize(
    "change", ["unpinned", "unsafe_path", "duplicate", "wrong_format", "commercial"]
)
def test_recipe_rejects_invalid_inputs(change):
    raw = CorpusBuildConfig.from_yaml(RECIPE).model_dump()
    if change == "unpinned":
        raw["sources"][0]["revision"] = "main"
    elif change == "unsafe_path":
        raw["sources"][0]["include"][0]["config"] = "../escape"
    elif change == "duplicate":
        raw["sources"].append(raw["sources"][0])
    elif change == "wrong_format":
        raw["sources"][0]["kind"] = "gifteval_pretrain"
    else:
        raw["commercial_use"] = True
    with pytest.raises(ValidationError):
        CorpusBuildConfig.model_validate(raw)
