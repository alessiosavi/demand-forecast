"""The batch index, rather than worker scheduling, addresses the stream."""

import importlib
import json
from ast import literal_eval
from copy import deepcopy
from dataclasses import fields, replace
from itertools import islice
from unittest.mock import DEFAULT, Mock, call

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, Item, Panel
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.schema import (
    KIND_CATEGORY,
    MAX_FEATURE_ROWS,
    N_SUMMARY,
    ROLE_PEER,
    SUMMARY_FEATURES,
)
from demand_forecast.data.episode.screening import screen
from demand_forecast.pretrain.v5 import t1
from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.pretrain.v5.corpus import CorpusValueError, ShortRow
from demand_forecast.pretrain.v5.sources import SourceMix, TypedSource
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
from tests.helpers.batch_fingerprint import assert_fingerprint
from tests.helpers.corpus_v4_fixture import write_rows
from tests.helpers.v5_sampler import FixedSource, config, equal_batches, fixed_mix, panel


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.sampler")


def test_batch_is_pure():
    c = config()
    a, ra = api().make_batch(c, 3)
    b, rb = api().make_batch(c, 3)
    equal_batches(a, b)
    assert ra == rb and ra["episodes"] == 4 and ra["supervised"] > 0


# Batch index 3 at 508a326. Floating arrays use the helper's tolerant summaries.
_COLD_BASE_FINGERPRINTS = {
    7: json.loads("""
{
    "x": {"dtype": "<f4", "shape": [2, 3, 4, 16, 39], "nan": 0, "sums": [1424.219102236093, 1851.7186012727907, 737.0235014183104]},
    "patch_valid": {"sha256": "95afa6bd44943e23d7f9c329bf3d98c1ab4e44b1083308a8e47d8acd9dc699fd"},
    "row_kind": {"sha256": "a3f6cd9e680eed19970eb58897d27402f5ea7d6b53b20da59aea24254d658f42"},
    "row_role": {"sha256": "1e98e879abf9a2b53079f95ff74cc972355e2eb5950222aac669dd607e38c092"},
    "row_names": {"sha256": "0a9a677a4b7caa83630d395cedef1e91f14213a2978258dfaef817a27a28dc60"},
    "row_summary": {"dtype": "<f4", "shape": [2, 3, 28], "nan": 0, "sums": [9.012979259714484, 23.707878237590194, 3.537778970175644]},
    "row_summary_flags": {"sha256": "1b2ceb9d965e7f68f1216406489f8e4f1336ed08585cd1dbfbf5a326851417a6"},
    "row_score": {"dtype": "<f4", "shape": [2, 3], "nan": 0, "sums": [4.05058066546917, 4.05058066546917, 2.0355967332919436]},
    "row_cand": {"sha256": "020c4665397abd9245b2d798424cbdf0904dbf76d2ab7e090df25da50da677fb"},
    "forecast_rows": {"sha256": "8b2a91af6870eeaa9471a55b682f27a05cdebf042c542d42e494896353899d10"},
    "keep": {"sha256": "6ace9e1c9c001921a3649a9e0840f93f6931745fed9495712302cf42e1bfa497"},
    "labels": {"dtype": "<f4", "shape": [2, 1, 1, 16], "nan": 0, "sums": [0.4499250443186611, 22.014989951392636, 2.7879090781061677]},
    "label_mask": {"sha256": "8e9562682e534552507f03538f76e2281d9011a9165ece88f8dd95ba1c72e1f9"},
    "loc": {"dtype": "<f8", "shape": [2, 1], "nan": 0, "sums": [116.44840613513641, 116.44840613513641, 112.65383921994614]},
    "scale": {"dtype": "<f8", "shape": [2, 1], "nan": 0, "sums": [55.51352855564641, 55.51352855564641, 54.037619391295564]},
    "fallback": {"sha256": "fbfde07e36e7b3d3d4ddb21a47f907a0dd9450c3d1a93c6dbb4a813ffdb2e149"},
    "known_rows": {"sha256": "63730b9ffa8f008a6a0b206d07dcf6e99f6e0d8361576906fb3a0707a5f4206c"},
    "xh": {"dtype": "<f4", "shape": [2, 1, 0, 1, 16, 39], "nan": 0, "sums": [0.0, 0.0, 0.0]},
    "cal_phases": {"dtype": "<f4", "shape": [2, 4, 16, 16], "nan": 0, "sums": [-449.4098955914378, 803.2651589997113, -209.9487302859343]},
    "cal_ids": {"sha256": "280ed3db0504778b758e282b0fca6ba68fb6dd64954fa9df4b35e340138ba00e"},
    "calh_phases": {"dtype": "<f4", "shape": [2, 1, 1, 16, 16], "nan": 0, "sums": [-73.65367571264505, 177.6252521649003, -35.182470519212075]},
    "calh_ids": {"sha256": "38bc19a775baff77b54f98eeb25cac7a60805d22ca9b0d72318fd0ce8407f51f"},
    "cand_names": {"sha256": "8ed2852fa2229720b58266608cef13729cad0d7ae0ee8c272c46c7940c31e4e0"},
    "cand_kind": {"sha256": "ec61fc8472f8dd3db3098a181e9797eea7a0edc70c3ab0e7db7f316ea5e42577"},
    "cand_summary": {"dtype": "<f4", "shape": [2, 1, 28], "nan": 0, "sums": [10.136119433678687, 42.27769184950739, 4.978625895056342]},
    "cand_flags": {"sha256": "29bb065446799325cd5c81804c7b39ea16a43f8ab4b239391702de0114350234"},
    "cand_score": {"dtype": "<f4", "shape": [2, 1], "nan": 0, "sums": [1.2005667686462402, 1.2005667686462402, 0.9004250764846802]},
    "attr_stats": {"dtype": "<f4", "shape": [2, 1, 57], "nan": 0, "sums": [46.30832657311112, 78.44989898893982, 27.725162405553352]},
    "attr_flags": {"sha256": "4c07f09597417753080b94ec18bf00031a72d67002b8722b94370b5a13868d30"},
    "attr_cand": {"sha256": "712245c87bc23c0d392e0e77e1806d5aa012ddcd22b8d3c0459c7650df06a5fc"},
    "attr_names": {"sha256": "6488d3eaf92ddd23b015859f69e029d713c81ccbb827fbe06dbf8e94cc1da3e1"},
    "static_num": {"dtype": "<f4", "shape": [2, 0], "nan": 0, "sums": [0.0, 0.0, 0.0]},
    "static_num_flags": {"sha256": "cd3d6d5bbd792b75d8a0b8e9ee7d8cc98612f58af73b5273e4f87997f937ef1a"},
    "static_num_cand": {"sha256": "07391ee9c6db75d1dd0916911fdf30a76a813f3f9f318d3f12c67b0536a7040e"},
    "freq_class": {"sha256": "aea8703e8437caebcb69bca5317b6b1299e25364a9bcb61197e78b9e6b0e1973"},
    "horizon": {"sha256": "edfaf6e16a71913b1a3dbaf89fcadcf7bf9a74f80c7c6273096dc5945a775ede"},
    "origin_patches": {"sha256": "9a7ec8661b2678a2837580c207c72f657ed7acd818f7f7dbcb3bac143722a43c"},
    "context_target": {"sha256": "290f3752350d63f6c3714b5d8ac6699e3492b24a757f9a3ab0339c0b06be6539"},
    "row_mask": {"sha256": "3f47a31804c39fc49c1c9efe7445e91beece44cd3e3b9e922dcf6c47294bb89d"},
    "forecast_mask": {"sha256": "640c6a26af7b94ebbe74627c779c3dd0342f216664f04852123ca1890a140ffe"},
    "known_mask": {"sha256": "ada5dbbd63796b07d679cece7a752730263e8a833a343c8ad5779546613a8395"},
    "cand_mask": {"sha256": "7431b585079924604b9100409e15bda1e4fb04927cb9b4959273e9383a8b27c6"},
    "attr_mask": {"sha256": "9f456d02994ace0c383c295ae9f3f0bfb379df56aa30b8e5ad935a163a526fe7"},
    "static_num_mask": {"sha256": "67d5149cf6f7003d83efeb558ad39b8f5d0e2a6351d573e78c8ea562d497d80b"},
    "origin_mask": {"sha256": "bbe08fe912f325e568aadccdbf6a0dff8e7586044c40fcb3847dacac1827e924"},
    "horizon_mask": {"sha256": "8ae6ee72f134a2600033a027ab1aef5fea56e49159c3da8f2c856164b440cb53"}
}
"""),
    12: json.loads("""
{
    "x": {"dtype": "<f4", "shape": [2, 3, 4, 16, 39], "nan": 0, "sums": [1666.9372142059729, 2200.4786044349894, 814.1164096445025]},
    "patch_valid": {"sha256": "b866507832ea435e5af944b1ea40fb0c91bc82b0d457155c3e2094fbcfffd666"},
    "row_kind": {"sha256": "a3f6cd9e680eed19970eb58897d27402f5ea7d6b53b20da59aea24254d658f42"},
    "row_role": {"sha256": "1e98e879abf9a2b53079f95ff74cc972355e2eb5950222aac669dd607e38c092"},
    "row_names": {"sha256": "bdef2bea9cfe148e65178995089ec6385e09107b8f2681bf2ebdf06a03de25c2"},
    "row_summary": {"dtype": "<f4", "shape": [2, 3, 28], "nan": 0, "sums": [16.21811730414629, 35.94975741952658, 4.912792791612446]},
    "row_summary_flags": {"sha256": "1b2ceb9d965e7f68f1216406489f8e4f1336ed08585cd1dbfbf5a326851417a6"},
    "row_score": {"dtype": "<f4", "shape": [2, 3], "nan": 0, "sums": [4.571596175432205, 4.571596175432205, 2.4694478611151376]},
    "row_cand": {"sha256": "020c4665397abd9245b2d798424cbdf0904dbf76d2ab7e090df25da50da677fb"},
    "forecast_rows": {"sha256": "8b2a91af6870eeaa9471a55b682f27a05cdebf042c542d42e494896353899d10"},
    "keep": {"sha256": "6ace9e1c9c001921a3649a9e0840f93f6931745fed9495712302cf42e1bfa497"},
    "labels": {"dtype": "<f4", "shape": [2, 1, 1, 16], "nan": 0, "sums": [4.586226940155029, 7.704958438873291, 0.6213695351034403]},
    "label_mask": {"sha256": "09b726b5195cfc4de7cb62c6bc6e4beac27b59bddf5cb7ac50ee2b445404d518"},
    "loc": {"dtype": "<f8", "shape": [2, 1], "nan": 0, "sums": [4.2911928079199, 4.2911928079199, 2.7144746558567983]},
    "scale": {"dtype": "<f8", "shape": [2, 1], "nan": 0, "sums": [1.3983661187396565, 1.3983661187396565, 0.8223754054451353]},
    "fallback": {"sha256": "fbfde07e36e7b3d3d4ddb21a47f907a0dd9450c3d1a93c6dbb4a813ffdb2e149"},
    "known_rows": {"sha256": "63730b9ffa8f008a6a0b206d07dcf6e99f6e0d8361576906fb3a0707a5f4206c"},
    "xh": {"dtype": "<f4", "shape": [2, 1, 0, 1, 16, 39], "nan": 0, "sums": [0.0, 0.0, 0.0]},
    "cal_phases": {"dtype": "<f4", "shape": [2, 4, 16, 16], "nan": 0, "sums": [-149.68964175134897, 810.2480550762266, -109.44892658174649]},
    "cal_ids": {"sha256": "8b3c3580a86c7cdd1dddec4f721e7692867edaedd05db1f53f703552eff4330a"},
    "calh_phases": {"dtype": "<f4", "shape": [2, 1, 1, 16, 16], "nan": 0, "sums": [30.36466570198536, 152.4492011219263, 11.320530368349864]},
    "calh_ids": {"sha256": "f75c6d2bf08e62be40b02cbd3cadb991ac0086b8a016da870a71ddb6269e3b6c"},
    "cand_names": {"sha256": "780550f099540085b8bb39fb65cc8aed2c1a465358e464c6e0f9989e53152ca6"},
    "cand_kind": {"sha256": "ec61fc8472f8dd3db3098a181e9797eea7a0edc70c3ab0e7db7f316ea5e42577"},
    "cand_summary": {"dtype": "<f4", "shape": [2, 1, 28], "nan": 0, "sums": [11.930494380998425, 39.564147200551815, 5.737912122133172]},
    "cand_flags": {"sha256": "29bb065446799325cd5c81804c7b39ea16a43f8ab4b239391702de0114350234"},
    "cand_score": {"dtype": "<f4", "shape": [2, 1], "nan": 0, "sums": [1.2369352579116821, 1.2369352579116821, 0.9616312682628632]},
    "attr_stats": {"dtype": "<f4", "shape": [2, 1, 57], "nan": 0, "sums": [50.981629058602266, 78.61528187815566, 31.167058409606273]},
    "attr_flags": {"sha256": "4c07f09597417753080b94ec18bf00031a72d67002b8722b94370b5a13868d30"},
    "attr_cand": {"sha256": "712245c87bc23c0d392e0e77e1806d5aa012ddcd22b8d3c0459c7650df06a5fc"},
    "attr_names": {"sha256": "a9e78d1666d7d261b718f05aeb80e358c55046a175d1ad692091812d6c2b05a6"},
    "static_num": {"dtype": "<f4", "shape": [2, 0], "nan": 0, "sums": [0.0, 0.0, 0.0]},
    "static_num_flags": {"sha256": "cd3d6d5bbd792b75d8a0b8e9ee7d8cc98612f58af73b5273e4f87997f937ef1a"},
    "static_num_cand": {"sha256": "07391ee9c6db75d1dd0916911fdf30a76a813f3f9f318d3f12c67b0536a7040e"},
    "freq_class": {"sha256": "aea8703e8437caebcb69bca5317b6b1299e25364a9bcb61197e78b9e6b0e1973"},
    "horizon": {"sha256": "e64e717b6c0edb728487a86d75e53ebb310e2e6dbf6e7bd2f802f7eb94741db9"},
    "origin_patches": {"sha256": "9a7ec8661b2678a2837580c207c72f657ed7acd818f7f7dbcb3bac143722a43c"},
    "context_target": {"sha256": "290f3752350d63f6c3714b5d8ac6699e3492b24a757f9a3ab0339c0b06be6539"},
    "row_mask": {"sha256": "3f47a31804c39fc49c1c9efe7445e91beece44cd3e3b9e922dcf6c47294bb89d"},
    "forecast_mask": {"sha256": "640c6a26af7b94ebbe74627c779c3dd0342f216664f04852123ca1890a140ffe"},
    "known_mask": {"sha256": "ada5dbbd63796b07d679cece7a752730263e8a833a343c8ad5779546613a8395"},
    "cand_mask": {"sha256": "7431b585079924604b9100409e15bda1e4fb04927cb9b4959273e9383a8b27c6"},
    "attr_mask": {"sha256": "9f456d02994ace0c383c295ae9f3f0bfb379df56aa30b8e5ad935a163a526fe7"},
    "static_num_mask": {"sha256": "67d5149cf6f7003d83efeb558ad39b8f5d0e2a6351d573e78c8ea562d497d80b"},
    "origin_mask": {"sha256": "bbe08fe912f325e568aadccdbf6a0dff8e7586044c40fcb3847dacac1827e924"},
    "horizon_mask": {"sha256": "8ae6ee72f134a2600033a027ab1aef5fea56e49159c3da8f2c856164b440cb53"}
}
"""),
    31: json.loads("""
{
    "x": {"dtype": "<f4", "shape": [2, 3, 4, 16, 39], "nan": 0, "sums": [1837.2535877772607, 2680.1371777676977, 929.4237397062727]},
    "patch_valid": {"sha256": "b866507832ea435e5af944b1ea40fb0c91bc82b0d457155c3e2094fbcfffd666"},
    "row_kind": {"sha256": "a3f6cd9e680eed19970eb58897d27402f5ea7d6b53b20da59aea24254d658f42"},
    "row_role": {"sha256": "1e98e879abf9a2b53079f95ff74cc972355e2eb5950222aac669dd607e38c092"},
    "row_names": {"sha256": "103f7c708246f45788f97f27152ff1bbd3ffb39da8f3991306cf78f3c7cdf2d0"},
    "row_summary": {"dtype": "<f4", "shape": [2, 3, 28], "nan": 0, "sums": [12.088519237426226, 38.82740180888504, 5.6667011196127755]},
    "row_summary_flags": {"sha256": "1b2ceb9d965e7f68f1216406489f8e4f1336ed08585cd1dbfbf5a326851417a6"},
    "row_score": {"dtype": "<f4", "shape": [2, 3], "nan": 0, "sums": [4.087208420038223, 4.087208420038223, 2.147775575518608]},
    "row_cand": {"sha256": "020c4665397abd9245b2d798424cbdf0904dbf76d2ab7e090df25da50da677fb"},
    "forecast_rows": {"sha256": "8b2a91af6870eeaa9471a55b682f27a05cdebf042c542d42e494896353899d10"},
    "keep": {"sha256": "6ace9e1c9c001921a3649a9e0840f93f6931745fed9495712302cf42e1bfa497"},
    "labels": {"dtype": "<f4", "shape": [2, 1, 1, 16], "nan": 0, "sums": [-0.30631808936595917, 32.34649269282818, -0.8994014305062592]},
    "label_mask": {"sha256": "8086fc4ccc0b40324a7d64c71ef7dea5facd2e1b52b8b2f5df5be89dabaa1088"},
    "loc": {"dtype": "<f8", "shape": [2, 1], "nan": 0, "sums": [993.4993192740998, 993.4993192740998, 991.5426981747759]},
    "scale": {"dtype": "<f8", "shape": [2, 1], "nan": 0, "sums": [106.66481790810698, 106.66481790810698, 106.37799228662256]},
    "fallback": {"sha256": "fbfde07e36e7b3d3d4ddb21a47f907a0dd9450c3d1a93c6dbb4a813ffdb2e149"},
    "known_rows": {"sha256": "63730b9ffa8f008a6a0b206d07dcf6e99f6e0d8361576906fb3a0707a5f4206c"},
    "xh": {"dtype": "<f4", "shape": [2, 1, 0, 1, 16, 39], "nan": 0, "sums": [0.0, 0.0, 0.0]},
    "cal_phases": {"dtype": "<f4", "shape": [2, 4, 16, 16], "nan": 0, "sums": [-357.0515031889081, 821.2261561807245, -199.9245217142161]},
    "cal_ids": {"sha256": "521caa502f5fab76ef865aa4372a8bed66e016637fa0705c74667be7d6b9d875"},
    "calh_phases": {"dtype": "<f4", "shape": [2, 1, 1, 16, 16], "nan": 0, "sums": [-30.77334886044264, 146.0799094364047, -21.814883736136835]},
    "calh_ids": {"sha256": "938cf4a3ba1acbde1d1d85ae2759d12b44b482b2dcfae833d0f159c906b21bf7"},
    "cand_names": {"sha256": "d2957d00eb7ecad632fc659b55c2b83b7b4d99e2eed429e56d18be54147b0c9f"},
    "cand_kind": {"sha256": "ec61fc8472f8dd3db3098a181e9797eea7a0edc70c3ab0e7db7f316ea5e42577"},
    "cand_summary": {"dtype": "<f4", "shape": [2, 1, 28], "nan": 0, "sums": [14.800240138079971, 64.71682283794507, 5.684248237571281]},
    "cand_flags": {"sha256": "29bb065446799325cd5c81804c7b39ea16a43f8ab4b239391702de0114350234"},
    "cand_score": {"dtype": "<f4", "shape": [2, 1], "nan": 0, "sums": [1.3472567200660706, 1.3472567200660706, 0.9285810589790344]},
    "attr_stats": {"dtype": "<f4", "shape": [2, 1, 57], "nan": 0, "sums": [62.21064047096297, 112.12722317082807, 34.75149635248409]},
    "attr_flags": {"sha256": "4c07f09597417753080b94ec18bf00031a72d67002b8722b94370b5a13868d30"},
    "attr_cand": {"sha256": "712245c87bc23c0d392e0e77e1806d5aa012ddcd22b8d3c0459c7650df06a5fc"},
    "attr_names": {"sha256": "1484d0b8ce5757268c093fa12f29359a4c1fb71ca8614ccff08abbc5f5a05581"},
    "static_num": {"dtype": "<f4", "shape": [2, 0], "nan": 0, "sums": [0.0, 0.0, 0.0]},
    "static_num_flags": {"sha256": "cd3d6d5bbd792b75d8a0b8e9ee7d8cc98612f58af73b5273e4f87997f937ef1a"},
    "static_num_cand": {"sha256": "07391ee9c6db75d1dd0916911fdf30a76a813f3f9f318d3f12c67b0536a7040e"},
    "freq_class": {"sha256": "aea8703e8437caebcb69bca5317b6b1299e25364a9bcb61197e78b9e6b0e1973"},
    "horizon": {"sha256": "52c9d465baa469dea824e26440d3da1ce0c2bde528df3358ef5039c0ec22498a"},
    "origin_patches": {"sha256": "9a7ec8661b2678a2837580c207c72f657ed7acd818f7f7dbcb3bac143722a43c"},
    "context_target": {"sha256": "290f3752350d63f6c3714b5d8ac6699e3492b24a757f9a3ab0339c0b06be6539"},
    "row_mask": {"sha256": "3f47a31804c39fc49c1c9efe7445e91beece44cd3e3b9e922dcf6c47294bb89d"},
    "forecast_mask": {"sha256": "640c6a26af7b94ebbe74627c779c3dd0342f216664f04852123ca1890a140ffe"},
    "known_mask": {"sha256": "ada5dbbd63796b07d679cece7a752730263e8a833a343c8ad5779546613a8395"},
    "cand_mask": {"sha256": "7431b585079924604b9100409e15bda1e4fb04927cb9b4959273e9383a8b27c6"},
    "attr_mask": {"sha256": "9f456d02994ace0c383c295ae9f3f0bfb379df56aa30b8e5ad935a163a526fe7"},
    "static_num_mask": {"sha256": "67d5149cf6f7003d83efeb558ad39b8f5d0e2a6351d573e78c8ea562d497d80b"},
    "origin_mask": {"sha256": "bbe08fe912f325e568aadccdbf6a0dff8e7586044c40fcb3847dacac1827e924"},
    "horizon_mask": {"sha256": "8ae6ee72f134a2600033a027ab1aef5fea56e49159c3da8f2c856164b440cb53"}
}
"""),
}

_COLD_BASE_REPORT = {
    "episodes": 2,
    "supervised": 19,
    "cells": 416,
    "skips": {},
    "sources": {"typed": 2},
    "paths": {"static_category": 2},
    "paths_without_candidate": {},
}


# Captured from the default batch after verifying the original byte pin.
_BASE_BATCH_FINGERPRINT = {
    "x": {
        "dtype": "<f4",
        "shape": [4, 5, 4, 16, 39],
        "nan": 0,
        "sums": [7603.942322080315, 10569.192989855248, 2981.167578761939],
    },
    "patch_valid": {"sha256": "40d4295b1d046b32fb1512afa1a246abd391d3c228c1744d1f0f9233312117c3"},
    "row_kind": {"sha256": "c50f83c4be15cc4026d790456c4a7dc76dc33b9397674dccc9fcc8ddac364d65"},
    "row_role": {"sha256": "f3b7e1440151dd1d9bab4f205e17532c648f7c20475c1246cba93cdb71b45cad"},
    "row_names": {"sha256": "f48341a6d92d0bd4066b0bc9eb5f6ad0b1e79e69da8c28ce34645a0e1c3bd171"},
    "row_summary": {
        "dtype": "<f4",
        "shape": [4, 5, 28],
        "nan": 0,
        "sums": [24.95725216716528, 66.92101139947772, 7.296810248700371],
    },
    "row_summary_flags": {
        "sha256": "0496f87a661e46b03776acc528e75b7fe583fa3486dc97924c765324ea372667"
    },
    "row_score": {
        "dtype": "<f4",
        "shape": [4, 5],
        "nan": 0,
        "sums": [9.71929798531346, 9.71929798531346, 4.004607424407731],
    },
    "row_cand": {"sha256": "cfba54ed55e48887711a3df2102e3455733cabcb0e03b7879b8ae4ce2dc7cd4d"},
    "forecast_rows": {"sha256": "d9b405550f8ba937554a2debb8cedcfc60819e43bf46ce546fe3de5ea85480f9"},
    "keep": {"sha256": "c33098e12386fa12ccf1c095fb5c2970b6fc77d81c77049e31446fda97c48118"},
    "labels": {
        "dtype": "<f4",
        "shape": [4, 2, 1, 16],
        "nan": 0,
        "sums": [-36.73539087176323, 70.0549784898758, -21.345336115686223],
    },
    "label_mask": {"sha256": "867726bc2b4795f845b0fa02f78ceb93ec3981d917fe3bf6fd62fa1c2f0873c4"},
    "loc": {
        "dtype": "<f8",
        "shape": [4, 1],
        "nan": 0,
        "sums": [665.2580041191961, 665.2580041191961, 565.2572249599358],
    },
    "scale": {
        "dtype": "<f8",
        "shape": [4, 1],
        "nan": 0,
        "sums": [487.8132051421306, 487.8132051421306, 448.84613740597064],
    },
    "fallback": {"sha256": "db7efe99401db002249bb62c50516312e56b7603670e2ca82c65b4b70b2a7b14"},
    "known_rows": {"sha256": "32019a29d4fabbce5d372e432300c2821923c06bb3b8066be1eff088033f5583"},
    "xh": {
        "dtype": "<f4",
        "shape": [4, 2, 1, 1, 16, 39],
        "nan": 0,
        "sums": [75.53048648685217, 82.93639671802521, 35.76268566896817],
    },
    "cal_phases": {
        "dtype": "<f4",
        "shape": [4, 4, 16, 16],
        "nan": 0,
        "sums": [-967.6367303989828, 1622.4486486576498, -482.09265684501406],
    },
    "cal_ids": {"sha256": "ffa9852dda0b1fc88af13e492109a4f17a081a62d8d7b9b9c6a3b6a80f2a1117"},
    "calh_phases": {
        "dtype": "<f4",
        "shape": [4, 2, 1, 16, 16],
        "nan": 0,
        "sums": [-126.77930046990514, 443.5837578885257, -57.88633335596023],
    },
    "calh_ids": {"sha256": "a8150d68fe149b463367dd739c2148da0ab4cce13f11fe5aa093616377e372b7"},
    "cand_names": {"sha256": "8588bf453a5cf737f6517f53971c2e3fe5ae8fd438e6d53936dd2f30399355cb"},
    "cand_kind": {"sha256": "0176a3baacf9e4bee69f8e17fe9efb7c5ff0d2a2fb441fa0f5df5989949c92f4"},
    "cand_summary": {
        "dtype": "<f4",
        "shape": [4, 2, 28],
        "nan": 0,
        "sums": [17.524523993663024, 57.61646507837577, 6.026995040776196],
    },
    "cand_flags": {"sha256": "5165afe560216db57c679ea14b1c3c907809ed942e530ae8d2d49ecc11deaa81"},
    "cand_score": {
        "dtype": "<f4",
        "shape": [4, 2],
        "nan": 0,
        "sums": [5.909398555755615, 5.909398555755615, 3.2973566353321075],
    },
    "attr_stats": {
        "dtype": "<f4",
        "shape": [4, 2, 57],
        "nan": 0,
        "sums": [28.157332506971784, 68.24927359168453, 7.738828159630274],
    },
    "attr_flags": {"sha256": "372fed7dfd6774f17a890fa9260d16935780be8c889bbf9e7c7a16730ed21306"},
    "attr_cand": {"sha256": "f2b2afad083e2d21c3dde2c55099624041b97310a87d6b293e791169bf55c7ed"},
    "attr_names": {"sha256": "05269ac5c67c028eb8a73b1cfbd65eeb9ac4f0ac390d70d62761d61d376da9c6"},
    "static_num": {
        "dtype": "<f4",
        "shape": [4, 2],
        "nan": 0,
        "sums": [-0.09850254654884338, 7.898325473070145, -1.2052506729960442],
    },
    "static_num_flags": {
        "sha256": "33731f74ef5f96de17cdcd3ad9302827adfbe62f8c6f037d434467470296a709"
    },
    "static_num_cand": {
        "sha256": "bc907162f3924eb228c4f88004159039bdac4bf41dbd7f9f01300bd9002d04d0"
    },
    "freq_class": {"sha256": "4f6d44463ea6c07f86e34e7367ebaf4834bca9843f886f0a60fb4aef93a6e10f"},
    "horizon": {"sha256": "12ce5f5f1732e2a6e3571b22c5fa4c728242f10f9a9a5c683a3795352a00a374"},
    "origin_patches": {
        "sha256": "490e558305cc4cbaef890f0692e7e5112e7b041a02df2e469ac6a32d5d7701eb"
    },
    "context_target": {
        "sha256": "60ecf51984c70257e47f21dcf6a885d7812adddfadd74c117730392618baaeb3"
    },
    "row_mask": {"sha256": "5622fa40b9b470762a613cae820435b282b1ce533a502db465791f63d56be3dc"},
    "forecast_mask": {"sha256": "4a2bf737d8509d9b4f7a0351c09a53e7b3cbfea3ea6d893012b246bfa6f02de1"},
    "known_mask": {"sha256": "6cc3a4279a316b4b2e867b340db06d1bfd36293fb34d92b037edeb82883c396d"},
    "cand_mask": {"sha256": "3f0c88411489f5672257c039c0cfe718ada438254cb10b472ab4189bc00ede60"},
    "attr_mask": {"sha256": "35a862cd7dcd9fd0ddfda9c54a9aa7d81dde7a7c54992e9be3a38a7952539a3d"},
    "static_num_mask": {
        "sha256": "64301f73f7cc57f8325cc83b23ef58ab4efbf8dc9cb4d1fd27b22cc81571c749"
    },
    "origin_mask": {"sha256": "dd14965ba984fd76d9d65a48e090f399d9c254a0755c404dde0e000767d67bcd"},
    "horizon_mask": {"sha256": "8a39239b3f6cea9702fa8ec16b8b9d918adb0e6ce07bfa140259cf994028a67e"},
}


def test_default_level_reproduces_the_base_batch():
    c = replace(config(drop=()), seed=12)
    source = replace(
        c.sources[0],
        recipes=(("static_number", "known_driver"),),
        max_items=20,
        length=(128, 160),
        n_nuisance=1,
        nuisance_from=("static_number",),
    )
    batch, _ = api().make_batch(replace(c, sources=(source,)), 3)
    assert any(batch.cand_names)
    assert_fingerprint(batch, _BASE_BATCH_FINGERPRINT)


# Captured from B0 (e3c7847) at batch index 3.
_TYPED_BATCH_FINGERPRINTS = {
    7: {
        "x": {
            "dtype": "<f4",
            "shape": [4, 9, 4, 16, 39],
            "nan": 0,
            "sums": [19396.53918107853, 29046.190680616863, 9620.430639495375],
        },
        "patch_valid": {
            "sha256": "af827b918c4b7417c0ef13f0b0b9c7eb8cc9ba162c2077528f71623abefa1d05"
        },
        "row_kind": {"sha256": "63dc99cc248e2ef3feece964defc333806586f27e30a8e22e48e6957c8c489f0"},
        "row_role": {"sha256": "da5f303fbbd6adb399208effca31f2f3656dacc494d0b71c0f93adf5804b35f7"},
        "row_names": {"sha256": "a18760f68e6324b079dfb55fde474099e8a8deea92b39456029dc9e76e9ac4e0"},
        "row_summary": {
            "dtype": "<f4",
            "shape": [4, 9, 28],
            "nan": 0,
            "sums": [127.40447885579954, 247.90684129660542, 58.06947219602565],
        },
        "row_summary_flags": {
            "sha256": "2b849180d7ba01e5b7e1ee51b654564a0478f9b0168fc52bbe78d7da10c5948d"
        },
        "row_score": {
            "dtype": "<f4",
            "shape": [4, 9],
            "nan": 0,
            "sums": [15.306854514870793, 15.306854514870793, 7.324924377866814],
        },
        "row_cand": {"sha256": "26d8f2f943ef5541afa4f4cf3f2aeb8f2ce3aae4973bfd93143dca02eda08f5c"},
        "forecast_rows": {
            "sha256": "39ac5b94a8686e76ce9f87349a9e2f222404b0f97e7f609b46e74e2a683d2cd0"
        },
        "keep": {"sha256": "cf9aefd9f86baf0db4101d3e7b68417843211bc86ea31a51c17163e7cc290ed1"},
        "labels": {
            "dtype": "<f4",
            "shape": [4, 2, 5, 16],
            "nan": 0,
            "sums": [77.14924589637667, 729.9972201520577, 26.305520660972974],
        },
        "label_mask": {
            "sha256": "693d72cf7b1f573c0f73eea4f25eb8a5809a18d67bd5a6531eb59c077b615755"
        },
        "loc": {
            "dtype": "<f8",
            "shape": [4, 5],
            "nan": 0,
            "sums": [1015.2433469236812, 1015.2433469236812, 738.6047574318735],
        },
        "scale": {
            "dtype": "<f8",
            "shape": [4, 5],
            "nan": 0,
            "sums": [541.0119390646117, 541.0119390646117, 394.8205599534638],
        },
        "fallback": {"sha256": "b38cbef2397325e7f8c850f16ac1d71a8b582cb5af88968630948e5d66ee02b4"},
        "known_rows": {
            "sha256": "324cd9cfcad0a4705b1271550a65e2383dc0611e524472cf98e5d6a91e0d53b6"
        },
        "xh": {
            "dtype": "<f4",
            "shape": [4, 2, 1, 1, 16, 39],
            "nan": 0,
            "sums": [362.94604098051786, 431.38820081949234, 241.691073047183],
        },
        "cal_phases": {
            "dtype": "<f4",
            "shape": [4, 4, 16, 16],
            "nan": 0,
            "sums": [911.5140599012375, 1616.9271729439497, 471.11620431763004],
        },
        "cal_ids": {"sha256": "234c3cb08b6661f5e83213b9f3c00788498b9ffb30650bf590b7280e92723699"},
        "calh_phases": {
            "dtype": "<f4",
            "shape": [4, 2, 1, 16, 16],
            "nan": 0,
            "sums": [331.1704882569611, 647.8136474899948, 168.8239853922205],
        },
        "calh_ids": {"sha256": "24b5f97b1482c9e5b676b549ee43aa9584a295be45e746900155cc9294b6288d"},
        "cand_names": {
            "sha256": "0b0250b72b3f2497c494a475936501aa3b6dc1ddfd4f242e8a0dd0c044002623"
        },
        "cand_kind": {"sha256": "1ae01015f75fa9e2ac4d7d4ce3548de24a744c739ad39385caf5b891030365d2"},
        "cand_summary": {
            "dtype": "<f4",
            "shape": [4, 2, 28],
            "nan": 0,
            "sums": [16.243196883238852, 49.60090250056237, 5.802565904300925],
        },
        "cand_flags": {
            "sha256": "0b27d08aa855015af8dfbbf144e6a3ab03dd8d340ec1e20203688a3ada98f896"
        },
        "cand_score": {
            "dtype": "<f4",
            "shape": [4, 2],
            "nan": 0,
            "sums": [5.690977215766907, 5.690977215766907, 2.8145853728055954],
        },
        "attr_stats": {
            "dtype": "<f4",
            "shape": [4, 2, 57],
            "nan": 0,
            "sums": [40.52110577421263, 73.87881139153615, 17.66579029246401],
        },
        "attr_flags": {
            "sha256": "445b7682af82a73124cffbe36c1d90787fccd76760ad2c9a1d8d3bb6ba34c670"
        },
        "attr_cand": {"sha256": "a2e7fa3c8946c88535d3f844d027a01d8676e2627509480d4aacb53af8ce658d"},
        "attr_names": {
            "sha256": "14043085843a113fc9c1264a5e0ed5afe17ca52731cb899cdd2ff45a32239d02"
        },
        "static_num": {
            "dtype": "<f4",
            "shape": [4, 1],
            "nan": 0,
            "sums": [2.8866843208670616, 4.39886698871851, 2.0989578422158957],
        },
        "static_num_flags": {
            "sha256": "46290e8317315218a4265903bfd22b0f6c2c1ca637dc28d07289a6de1294af88"
        },
        "static_num_cand": {
            "sha256": "15ed3d8c8d6953eed7004a67d51472b97db8abae61940fcbdf05cacbaf55b5a7"
        },
        "freq_class": {
            "sha256": "4f6d44463ea6c07f86e34e7367ebaf4834bca9843f886f0a60fb4aef93a6e10f"
        },
        "horizon": {"sha256": "0703502d9867da97e62b9aecd9d84e84d8f07757b3e6255bf22fb81e433d1d67"},
        "origin_patches": {
            "sha256": "8ff69dd49f9ef302f32d9a20b76af84143283d780df85309e25f552e5a5be25a"
        },
        "context_target": {
            "sha256": "2b7d3821b8fba0a639212dd4f141e039c9125900db0ff6f55dd34d862073049a"
        },
        "row_mask": {"sha256": "022a515b342b3492954900bad91df3903116567397c7c828aa53083cb5654c4f"},
        "forecast_mask": {
            "sha256": "54f22486803fdb50a8c47ab040963831d46ddc9fa6545b04ab6089c4092ad63c"
        },
        "known_mask": {
            "sha256": "6cc3a4279a316b4b2e867b340db06d1bfd36293fb34d92b037edeb82883c396d"
        },
        "cand_mask": {"sha256": "e4c8a1f3580400ef42a732bc6b8c3978cbcd83d6c55538f2308dd7b97c6309f1"},
        "attr_mask": {"sha256": "2ca8a31b2fd7ce479672210dbc21de9a3168d6abf8c612ab66ee38f63014a31c"},
        "static_num_mask": {
            "sha256": "a04db465fb41c3747c725014ce6b6c2b9fcaf23d4c1cf800a706b3e65c8f298f"
        },
        "origin_mask": {
            "sha256": "2b89aab19449c850f20db0491162c18cb77bb7ba747d4073991bb921635bd1a3"
        },
        "horizon_mask": {
            "sha256": "8a39239b3f6cea9702fa8ec16b8b9d918adb0e6ce07bfa140259cf994028a67e"
        },
    },
    12: {
        "x": {
            "dtype": "<f4",
            "shape": [4, 6, 4, 16, 39],
            "nan": 0,
            "sums": [12170.637703122233, 17968.72251434218, 5591.447761481948],
        },
        "patch_valid": {
            "sha256": "da260cb16b20a661143f8fa26ccd58e14e7799c030d896ab09c7606d93053b4e"
        },
        "row_kind": {"sha256": "c699713d261c049b9d9bfca6fa787313570ea57c49f5b70120659e0e76cd9c33"},
        "row_role": {"sha256": "8d81ef99c9199ce650a81ecebe8519069db7c811491923d1b4532a6b32d395b6"},
        "row_names": {"sha256": "ba01b7afb0d13a6a3ddf20118b9f533be3602d9321eda9d3827475fef2a8c7c8"},
        "row_summary": {
            "dtype": "<f4",
            "shape": [4, 6, 28],
            "nan": 0,
            "sums": [41.43908091430785, 105.75558757857652, 19.816720014150114],
        },
        "row_summary_flags": {
            "sha256": "91c675fd185bd5b1a80a946778ff758a09e401aaec645cb31a0b410e2626ce59"
        },
        "row_score": {
            "dtype": "<f4",
            "shape": [4, 6],
            "nan": 0,
            "sums": [12.704358786344528, 12.704358786344528, 6.0659178933128715],
        },
        "row_cand": {"sha256": "0d46948c86d7d185a7144a2b0d38edeadbb372b4eec0d2729da6d10c9d1eb71b"},
        "forecast_rows": {
            "sha256": "b197245a06600cc0cf07043e472b368556348a4280654dca21d6d629a190fdc8"
        },
        "keep": {"sha256": "65c9e7ef494ad8638c1916ad03ec89c02fc9091a52f6c5ff2abd7bebb960a667"},
        "labels": {
            "dtype": "<f4",
            "shape": [4, 2, 2, 16],
            "nan": 0,
            "sums": [44.70124554913491, 216.6931647164747, 15.935887307136],
        },
        "label_mask": {
            "sha256": "9e43a7ba43babaa95a480444e55bc0975c4c42ab5c9e374916dfac3e60c40693"
        },
        "loc": {
            "dtype": "<f8",
            "shape": [4, 2],
            "nan": 0,
            "sums": [599.7619474165305, 599.7619474165305, 294.7752512039127],
        },
        "scale": {
            "dtype": "<f8",
            "shape": [4, 2],
            "nan": 0,
            "sums": [201.87215073040406, 201.87215073040406, 92.79969548300147],
        },
        "fallback": {"sha256": "0d6ca996b4cb75b54eeafb788df2524ae722b5abf74145106de81ed42b8f1e0d"},
        "known_rows": {
            "sha256": "053f1d2f4f09641c7c07f07b8bb8f818de63530ecbd02cd44d498534059b8fdb"
        },
        "xh": {
            "dtype": "<f4",
            "shape": [4, 2, 1, 1, 16, 39],
            "nan": 0,
            "sums": [324.5652963668108, 437.5685237199068, 173.91436889730332],
        },
        "cal_phases": {
            "dtype": "<f4",
            "shape": [4, 4, 16, 16],
            "nan": 0,
            "sums": [-930.524924069643, 1620.805704973638, -456.2568814872793],
        },
        "cal_ids": {"sha256": "453e7f0fe7d38c5d82c9bbc2f217d901334f22afeaaae8fb277b4a98997347f6"},
        "calh_phases": {
            "dtype": "<f4",
            "shape": [4, 2, 1, 16, 16],
            "nan": 0,
            "sums": [-166.87890073657036, 443.9682274758816, -78.12147868983448],
        },
        "calh_ids": {"sha256": "9b6cf514b8e7bef5cb7f506aa1e6a5bddec236f85110752a6b1e33a68036e626"},
        "cand_names": {
            "sha256": "c369fe9f86af5c02a42a8c7ca5754f95edd832d0b5ab27ffe711faacf06a6b1a"
        },
        "cand_kind": {"sha256": "93ac84741cec9491918b82c09b4599e326f17e6449dfa22f9247f4a6aad545bf"},
        "cand_summary": {
            "dtype": "<f4",
            "shape": [4, 2, 28],
            "nan": 0,
            "sums": [36.96759024867788, 103.73798943916336, 19.06447774133786],
        },
        "cand_flags": {
            "sha256": "c4e773b0174d0ef98d0d1da7eeb4cd8abd986b6940932cc6a4eb740f836a51dd"
        },
        "cand_score": {
            "dtype": "<f4",
            "shape": [4, 2],
            "nan": 0,
            "sums": [5.392868995666504, 5.392868995666504, 3.006906747817993],
        },
        "attr_stats": {
            "dtype": "<f4",
            "shape": [4, 2, 57],
            "nan": 0,
            "sums": [84.97128011612222, 151.7416793066077, 46.03030277892187],
        },
        "attr_flags": {
            "sha256": "975634f0c4e258256bec49f609b9137acf04f1f60aa800294e61a32a87753803"
        },
        "attr_cand": {"sha256": "f2b2afad083e2d21c3dde2c55099624041b97310a87d6b293e791169bf55c7ed"},
        "attr_names": {
            "sha256": "c7e0b2b969650348c13ebe8ef85b96ada2e370d18cd4699077349c24f988fe1c"
        },
        "static_num": {
            "dtype": "<f4",
            "shape": [4, 1],
            "nan": 0,
            "sums": [-0.6181924939155579, 3.2670000195503235, -1.075422316789627],
        },
        "static_num_flags": {
            "sha256": "46290e8317315218a4265903bfd22b0f6c2c1ca637dc28d07289a6de1294af88"
        },
        "static_num_cand": {
            "sha256": "15ed3d8c8d6953eed7004a67d51472b97db8abae61940fcbdf05cacbaf55b5a7"
        },
        "freq_class": {
            "sha256": "4f6d44463ea6c07f86e34e7367ebaf4834bca9843f886f0a60fb4aef93a6e10f"
        },
        "horizon": {"sha256": "12ce5f5f1732e2a6e3571b22c5fa4c728242f10f9a9a5c683a3795352a00a374"},
        "origin_patches": {
            "sha256": "490e558305cc4cbaef890f0692e7e5112e7b041a02df2e469ac6a32d5d7701eb"
        },
        "context_target": {
            "sha256": "124e7d795c9c26136fbc840b34d6686ed7e864244207a2a983fc8ff751e96f61"
        },
        "row_mask": {"sha256": "12f868d733df3886ee345a1993af4fc71c22c3202e1522e153244deec812ac3b"},
        "forecast_mask": {
            "sha256": "44fdb20edc2a9ddbccbc60b8024329008bb349fa4322e3ac1b0f78ae989ec1fd"
        },
        "known_mask": {
            "sha256": "6cc3a4279a316b4b2e867b340db06d1bfd36293fb34d92b037edeb82883c396d"
        },
        "cand_mask": {"sha256": "3f0c88411489f5672257c039c0cfe718ada438254cb10b472ab4189bc00ede60"},
        "attr_mask": {"sha256": "35a862cd7dcd9fd0ddfda9c54a9aa7d81dde7a7c54992e9be3a38a7952539a3d"},
        "static_num_mask": {
            "sha256": "a04db465fb41c3747c725014ce6b6c2b9fcaf23d4c1cf800a706b3e65c8f298f"
        },
        "origin_mask": {
            "sha256": "dd14965ba984fd76d9d65a48e090f399d9c254a0755c404dde0e000767d67bcd"
        },
        "horizon_mask": {
            "sha256": "8a39239b3f6cea9702fa8ec16b8b9d918adb0e6ce07bfa140259cf994028a67e"
        },
    },
    31: {
        "x": {
            "dtype": "<f4",
            "shape": [4, 9, 4, 16, 39],
            "nan": 0,
            "sums": [26697.06472066752, 38438.82168665118, 12021.394733265126],
        },
        "patch_valid": {
            "sha256": "63c12f4b57553249af21f6a07761213d894a31626cbfb43fad927a8cb4a875d4"
        },
        "row_kind": {"sha256": "90751d5dcc8e40809c1d6497000d861c1915672f46eaf5dd442948b5ff4eae04"},
        "row_role": {"sha256": "983674c7c10fb295c1e762779c2199f067f73f95197a84cf5e3e1ea682996b24"},
        "row_names": {"sha256": "e3f98636e691fc581b1c443ad19cd1d47f00f2653fe2e9119d0b7c38ccfbce25"},
        "row_summary": {
            "dtype": "<f4",
            "shape": [4, 9, 28],
            "nan": 0,
            "sums": [200.49928768681457, 348.68740553738644, 84.19079969223822],
        },
        "row_summary_flags": {
            "sha256": "e383f69782027cee9aa4770c69406e2a23493ece37d6bc812cca3fc9a91c7508"
        },
        "row_score": {
            "dtype": "<f4",
            "shape": [4, 9],
            "nan": 0,
            "sums": [20.725667759776115, 20.725667759776115, 9.894935799141725],
        },
        "row_cand": {"sha256": "2252f349a3f49573645e77231a7734f5fd3479357e0a7b9d3f8e16ab2cf065e3"},
        "forecast_rows": {
            "sha256": "39ac5b94a8686e76ce9f87349a9e2f222404b0f97e7f609b46e74e2a683d2cd0"
        },
        "keep": {"sha256": "cf9aefd9f86baf0db4101d3e7b68417843211bc86ea31a51c17163e7cc290ed1"},
        "labels": {
            "dtype": "<f4",
            "shape": [4, 2, 5, 16],
            "nan": 0,
            "sums": [180.0185144865536, 766.2318811440491, 153.583212382451],
        },
        "label_mask": {
            "sha256": "b9ef165a594b1139225b1d2c24fa1af8f874e73a395aa091f9d201aa14fed4e6"
        },
        "loc": {
            "dtype": "<f8",
            "shape": [4, 5],
            "nan": 0,
            "sums": [3563.948597821008, 3719.130452493024, 1603.0820583708432],
        },
        "scale": {
            "dtype": "<f8",
            "shape": [4, 5],
            "nan": 0,
            "sums": [2198.792262998099, 2198.792262998099, 1076.5080319190126],
        },
        "fallback": {"sha256": "b38cbef2397325e7f8c850f16ac1d71a8b582cb5af88968630948e5d66ee02b4"},
        "known_rows": {
            "sha256": "66746f76b3ba79acfc230e51ec697805807e70d00e47251a89da4efe69977f34"
        },
        "xh": {"dtype": "<f4", "shape": [4, 2, 0, 1, 16, 39], "nan": 0, "sums": [0.0, 0.0, 0.0]},
        "cal_phases": {
            "dtype": "<f4",
            "shape": [4, 4, 16, 16],
            "nan": 0,
            "sums": [1032.5979373902082, 1668.9709471017122, 519.3168162798975],
        },
        "cal_ids": {"sha256": "eb276c759ef11e02e920650d13226466bebe4b0c293f65d88c4ae3f62d6b4970"},
        "calh_phases": {
            "dtype": "<f4",
            "shape": [4, 2, 1, 16, 16],
            "nan": 0,
            "sums": [352.13913783431053, 708.2974537909031, 167.20142331067473],
        },
        "calh_ids": {"sha256": "78e46b23f165f0d67a99a97119d9d8270c5ebe74ea4219550ef4cd8b1dd821dc"},
        "cand_names": {
            "sha256": "b5bd91855340a04d803b3983550db4be013035be1b6f2492fd691e189585455b"
        },
        "cand_kind": {"sha256": "93ac84741cec9491918b82c09b4599e326f17e6449dfa22f9247f4a6aad545bf"},
        "cand_summary": {
            "dtype": "<f4",
            "shape": [4, 2, 28],
            "nan": 0,
            "sums": [41.91528476991516, 108.03682615452271, 22.41188652811356],
        },
        "cand_flags": {
            "sha256": "e2aeaeeaaa8b3e44a250b08426cba2e81a0182235095c6bd2968ed5e0079308e"
        },
        "cand_score": {
            "dtype": "<f4",
            "shape": [4, 2],
            "nan": 0,
            "sums": [6.396966218948364, 6.396966218948364, 3.49292279034853],
        },
        "attr_stats": {
            "dtype": "<f4",
            "shape": [4, 2, 57],
            "nan": 0,
            "sums": [68.64149155559399, 134.76303294020153, 35.146249208918135],
        },
        "attr_flags": {
            "sha256": "c52a3288af2dbd481c9db118b3dc1cf059f320de7889d9d4af1a2c8232a8dbf8"
        },
        "attr_cand": {"sha256": "f2b2afad083e2d21c3dde2c55099624041b97310a87d6b293e791169bf55c7ed"},
        "attr_names": {
            "sha256": "6abd1283731ddc52d0458a7c31589315cd9e0633a16df8f77c02b216ad61c627"
        },
        "static_num": {
            "dtype": "<f4",
            "shape": [4, 1],
            "nan": 0,
            "sums": [2.888610363006592, 4.742579340934753, 2.938412308692932],
        },
        "static_num_flags": {
            "sha256": "46290e8317315218a4265903bfd22b0f6c2c1ca637dc28d07289a6de1294af88"
        },
        "static_num_cand": {
            "sha256": "15ed3d8c8d6953eed7004a67d51472b97db8abae61940fcbdf05cacbaf55b5a7"
        },
        "freq_class": {
            "sha256": "4f6d44463ea6c07f86e34e7367ebaf4834bca9843f886f0a60fb4aef93a6e10f"
        },
        "horizon": {"sha256": "0703502d9867da97e62b9aecd9d84e84d8f07757b3e6255bf22fb81e433d1d67"},
        "origin_patches": {
            "sha256": "490e558305cc4cbaef890f0692e7e5112e7b041a02df2e469ac6a32d5d7701eb"
        },
        "context_target": {
            "sha256": "219d832ef6a955c4f18a5fa65ed1a30a668d5ccba61681bae8d8ae6033687587"
        },
        "row_mask": {"sha256": "d60ba14eddd9d5b212cce9a125c6feed09be58b5646f51f49b90605f6b3bc546"},
        "forecast_mask": {
            "sha256": "54f22486803fdb50a8c47ab040963831d46ddc9fa6545b04ab6089c4092ad63c"
        },
        "known_mask": {
            "sha256": "ce8b1cc799efc0adb3848019e9720ba6f86c15aa1566186aa404b51ee3d74b52"
        },
        "cand_mask": {"sha256": "3f0c88411489f5672257c039c0cfe718ada438254cb10b472ab4189bc00ede60"},
        "attr_mask": {"sha256": "35a862cd7dcd9fd0ddfda9c54a9aa7d81dde7a7c54992e9be3a38a7952539a3d"},
        "static_num_mask": {
            "sha256": "a04db465fb41c3747c725014ce6b6c2b9fcaf23d4c1cf800a706b3e65c8f298f"
        },
        "origin_mask": {
            "sha256": "dd14965ba984fd76d9d65a48e090f399d9c254a0755c404dde0e000767d67bcd"
        },
        "horizon_mask": {
            "sha256": "8a39239b3f6cea9702fa8ec16b8b9d918adb0e6ce07bfa140259cf994028a67e"
        },
    },
}


@pytest.mark.parametrize(
    ("seed", "summary_off"),
    [(7, ()), (12, ()), (31, ("zero_share", "log_length"))],
)
def test_training_batches_are_unchanged(seed, summary_off):
    c = config(designate_target=True, drop=(), summary_off=summary_off)
    source = replace(
        c.sources[0],
        recipes=(("static_category", "static_number", "dynamic_category_mean", "multi_target"),),
        max_items=12,
        length=(128, 160),
    )
    batch, _ = api().make_batch(replace(c, sources=(source,), seed=seed), 3)
    assert (batch.forecast_mask.sum(dim=1) > 1).all()
    assert batch.cand_mask.any() and ((batch.row_kind == KIND_CATEGORY) & batch.row_mask).any()
    assert_fingerprint(batch, _TYPED_BATCH_FINGERPRINTS[seed])


@pytest.mark.parametrize("designate_target", [False, True])
def test_level_zeroes_scores_and_keeps_every_candidate_within_budget(designate_target):
    generated = generate_typed_panel(
        PanelRecipe(
            ("static_category",),
            n_items=20,
            length=(128, 160),
            n_nuisance=1,
            nuisance_from=("static_category",),
            complete_inputs=True,
            seed=2,
        )
    )
    nuisance = generated.inputs_of("nuisance")[0].name
    base = config(drop=(), designate_target=designate_target, cutoff_at_origin=1)
    sources = fixed_mix(base, FixedSource(generated.panel))
    level = 0.75
    cfg = replace(base, sampler=replace(base.sampler, screen_no_evidence=level))
    reference, _ = api().make_batch(base, 3, sources=sources)
    screened, _ = api().make_batch(cfg, 3, sources=sources)

    candidate_index = reference.cand_names[0].index(nuisance)
    row = next(name for name in reference.row_names[0] if name.startswith(f"{nuisance}="))
    assert 0 < reference.cand_score[0, candidate_index] < level
    assert 0 < reference.row_score[0, reference.row_names[0].index(row)] < level
    assert screened.cand_names == reference.cand_names
    assert screened.cand_score[0, candidate_index] == 0
    assert screened.row_score[0, screened.row_names[0].index(row)] == 0


def test_level_puts_candidates_without_evidence_last_when_the_budget_binds(monkeypatch):
    rng = np.random.default_rng(0)
    length = 160
    target = np.sin(np.arange(length) / 5)
    known = {f"k{i:02d}": rng.normal(size=length) for i in range(20)}
    known["k19"] = target + rng.normal(scale=0.03, size=length)
    data = Panel(
        "feature-budget",
        (Item("i", 0, {"y": target}, known, {}),),
        dict.fromkeys(known, "known"),
        "D",
        None,
        True,
    )
    base = config(
        drop=(), cutoff_at_origin=1, episodes_per_batch=1, episodes_per_panel=1, origins=1
    )
    sources = fixed_mix(base, FixedSource(data))
    requests = []
    build = EpisodeBuilder.build

    def observe(builder, request):
        requests.append(request)
        return build(builder, request)

    monkeypatch.setattr(EpisodeBuilder, "build", observe)
    level = 0.75
    cfg = replace(base, sampler=replace(base.sampler, screen_no_evidence=level))
    reference, _ = api().make_batch(base, 3, sources=sources)
    screened, _ = api().make_batch(cfg, 3, sources=sources)
    request, screened_request = requests
    assert screened_request == request
    reference_names = set(reference.row_names[0]) & known.keys()
    assert reference_names == set(
        "k01 k02 k03 k04 k05 k06 k08 k10 k11 k12 k14 k15 k16 k17 k18 k19".split()
    )

    stats = PanelStats(data, "y", request.cutoff)
    scores = {
        candidate.name: float(np.float32(candidate.score)) if candidate.score >= level else 0.0
        for candidate in screen(data, stats, request.item_index, ("y",), request.cutoff)
        if candidate.removed is None
    }
    names = screened.row_names[0]
    admitted = {
        name: score.item()
        for name, score in zip(names, screened.row_score[0, : len(names)], strict=True)
        if name in known
    }
    cut = scores.keys() - admitted.keys()
    assert len(scores) == len(known) > MAX_FEATURE_ROWS == len(admitted)
    assert admitted.keys() != reference_names
    assert admitted == {name: scores[name] for name in admitted}
    assert {scores[name] for name in cut} == {0.0}
    assert min(admitted.values()) >= max(scores[name] for name in cut)
    assert max(admitted.values()) >= level


@pytest.mark.parametrize("mechanism", ["static_category", "known_driver"])
def test_path_share_counts_kept_episodes_over_twenty_batches(mechanism):
    c = config(path_share=0.5, designate_target=True, episodes_per_panel=8)
    c = replace(
        c,
        sources=(replace(c.sources[0], recipes=((mechanism,),), max_items=24, length=(128, 160)),),
    )
    paths = kept = 0
    for k in range(20):
        _, report = api().make_batch(c, k)
        assert sum(report["paths"].values()) <= 2
        paths += sum(report["paths"].values())
        kept += report["episodes"]
    assert paths / kept >= 0.45
    if mechanism == "known_driver":
        assert paths / kept <= 0.55


@pytest.mark.parametrize("share", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_path_quota_is_never_exceeded(share):
    c = config(path_share=share, designate_target=True, episodes_per_batch=7, episodes_per_panel=8)
    c = replace(c, sources=(replace(c.sources[0], recipes=(("known_driver",),)),))
    for k in range(3):
        _, report = api().make_batch(c, k)
        assert report["episodes"] == 7
        assert sum(report["paths"].values()) == round(share * 7)


def test_typed_corpus_mix_fills_any_open_quota(tmp_path):
    pytest.importorskip("pyarrow")
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[np.arange(64, dtype=np.float64)]])
    c = config(path_share=1.0, designate_target=True)
    c = replace(
        c,
        sources=(
            replace(c.sources[0], recipes=(("known_driver",),)),
            SourceConfig("corpus", 1, root=str(tmp_path)),
        ),
    )
    seen_corpus = False
    for k in range(10):
        _, report = api().make_batch(c, k)
        assert report["episodes"] == 4
        assert sum(report["paths"].values()) <= 4
        seen_corpus |= report["sources"].get("corpus", 0) > 0
    assert seen_corpus


@pytest.mark.parametrize("mechanisms", [(), ("known_driver",)])
def test_plan_panel_respects_open_quotas(mechanisms):
    c = config(path_share=1.0, designate_target=True, episodes_per_panel=8)
    source = replace(c.sources[0], recipes=(mechanisms,))
    p, record = TypedSource(source).panel(np.random.default_rng(2))
    _, plans, _ = api().plan_panel(
        p, record, c.sampler, source, np.random.default_rng(3), path_quota=2, ordinary_quota=0
    )
    assert len(plans) == 2
    assert all(bool(q.mechanism) == bool(mechanisms) for q in plans)
    _, ordinary, _ = api().plan_panel(
        p, record, c.sampler, source, np.random.default_rng(3), path_quota=0, ordinary_quota=1
    )
    assert len(ordinary) == 1 and ordinary[0].mechanism is None


def test_failed_path_search_cannot_fill_closed_ordinary_quota():
    g = generate_typed_panel(PanelRecipe(("static_category",), n_items=24, seed=2))
    for item in g.panel.items:
        for name in item.static:
            item.static[name] = None
    c = config(path_share=1.0, designate_target=True)
    _, plans, skips = api().plan_panel(
        g.panel,
        {"source": "typed", "generated": g},
        c.sampler,
        c.sources[0],
        np.random.default_rng(3),
        path_quota=4,
        ordinary_quota=0,
    )
    assert not plans and skips["no_support"] == 1


@pytest.mark.parametrize("reason", ["short_row", "unsupervised", "oversized"])
def test_path_quota_replenishes_after_build_skip(monkeypatch, reason):
    c = config(path_share=0.5, designate_target=True, max_cells=100_000)
    c = replace(c, sources=(replace(c.sources[0], recipes=(("known_driver",),)),))
    original = api().EpisodeBuilder.build
    calls = 0

    def skip_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if reason == "short_row":
                raise ShortRow()
            if reason == "oversized":
                return shaped_episode(40, 160)
            episode = original(*args, **kwargs)
            episode.label_mask[:] = False
            return episode
        return original(*args, **kwargs)

    monkeypatch.setattr(api().EpisodeBuilder, "build", skip_once)
    _, report = api().make_batch(c, 0)
    assert report["episodes"] == 4 and report["paths"] == {"known_driver": 2}
    assert report["skips"][reason] == 1


@pytest.mark.parametrize("invalid", ["bare", "mapping", "object", "sequence"])
def test_make_batch_rejects_sources_other_than_source_mix(invalid):
    c = config()
    sources = {
        "bare": TypedSource(c.sources[0]),
        "mapping": {},
        "object": object(),
        "sequence": c.sources,
    }[invalid]
    with pytest.raises(TypeError, match="SourceMix"):
        api().make_batch(c, 0, sources=sources)


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_loader_resume_workers_and_default_torch_generator(workers, monkeypatch):
    c = config(episodes_per_batch=1)
    initial = torch.get_rng_state().clone()
    reference = list(islice(api().loader(c, 0), 9))
    real_loader = api().DataLoader

    def bounded_loader(*args, **kwargs):
        # A denied worker IPC operation must fail promptly, rather than hang CI.
        return real_loader(*args, timeout=10 if workers else 0, **kwargs)

    monkeypatch.setattr(api(), "DataLoader", bounded_loader)
    actual = list(islice(api().loader(replace(c, workers=workers), 3), 6))
    for (k, a, ra), (j, b, rb) in zip(reference[3:], actual, strict=True):
        assert k == j and ra == rb
        equal_batches(a, b)
    assert torch.equal(initial, torch.get_rng_state())


def static_batch_config(**sampler):
    defaults = dict(
        episodes_per_batch=2, episodes_per_panel=2, designate_target=True, path_share=1, drop=()
    )
    c = config(**(defaults | sampler))
    source = replace(c.sources[0], recipes=(("static_category",),), max_items=24, length=(128, 160))
    return replace(c, sources=(source,), seed=7)


@pytest.mark.parametrize("seed", [7, 12, 31])
def test_defaults_reproduce_the_base_batches(seed):
    batch, _ = api().make_batch(replace(static_batch_config(), seed=seed), 3)
    assert_fingerprint(batch, _COLD_BASE_FINGERPRINTS[seed])


@pytest.fixture
def cold_panel():
    generated = generate_typed_panel(PanelRecipe(("static_category",), n_items=4, seed=8))
    attribute = generated.inputs_of("static_category")[0].name
    items = []
    for i, (start, length, label) in enumerate(
        [(32, 32, "a"), (32, 32, "b"), (0, 96, "a"), (0, 96, "b")]
    ):
        values = np.arange(length, dtype=np.float64) + i
        items.append(
            Item(str(i), start, {"y0": values}, {"past": values.copy()}, {attribute: label})
        )
    p = Panel("cold", tuple(items), {"past": "past"}, None, None, True)
    return {
        "source": "typed",
        "generated": replace(generated, panel=p, complete=p, short_items=("0", "1")),
    }


def scripted_cold_rng(indices=(0,), steps=(3,), coins=(0.2,)):
    rng = Mock(wraps=np.random.default_rng(11))
    rng.choice.side_effect = [64, "static_category", *indices, "y0"]
    rng.integers.side_effect = [50 + i for i in range(len(indices))]
    rng.attach_mock(Mock(), "cold")
    rng.spawn.return_value = [rng.cold]
    rng.cold.random.side_effect = coins
    rng.cold.integers.side_effect = steps
    return rng


def plan_cold(record, rng, **sampler):
    c = static_batch_config(**({"cold_share": 1.0, "episodes_per_panel": 1} | sampler))
    return api().plan_panel(record["generated"].panel, record, c.sampler, c.sources[0], rng)


def test_cold_lead_frame(cold_panel):
    rng = scripted_cold_rng(steps=(4,))
    p, (plan,), counts = plan_cold(cold_panel, rng)
    request = plan.request
    item = p.items[request.item_index]
    assert plan.cold and plan.mechanism == "static_category"
    assert request.origins == (item.start + 4,) and request.cutoff == request.origins[0]
    values = item.targets[request.target]
    assert np.isfinite(values[:4]).any() and np.isfinite(values[4 : 4 + request.horizon]).any()
    episode = EpisodeBuilder(p).build(request)
    assert episode.label_mask.any()
    (attribute,) = item.static
    assert any(
        role == ROLE_PEER and name == f"{attribute}={item.static[attribute]}"
        for name, role in zip(episode.row_names, episode.row_role, strict=True)
    )
    assert np.isfinite(episode.loc).all()
    assert (np.isfinite(episode.scale) & (episode.scale > 0)).all()
    assert counts.get("fallback", 0) == 0
    rng.spawn.assert_called_once_with(1)
    rng.cold.random.assert_called_once_with()
    rng.cold.integers.assert_called_once_with(3, 5)


@pytest.mark.parametrize("constant", [False, True])
def test_cold_lead_with_a_missing_step(cold_panel, constant):
    values = cold_panel["generated"].panel.items[0].targets["y0"]
    values[:3] = [np.nan, 4, 4 if constant else 7]
    _, plans, counts = plan_cold(cold_panel, scripted_cold_rng())
    assert counts["cold_attempted"] == 1 and counts["cold_planned"] == int(not constant)
    assert counts.get("fallback", 0) == int(constant)
    if constant:
        assert not plans
    else:
        (plan,) = plans
        assert plan.cold and plan.request.origins == (35,)


def test_cold_does_not_move_the_other_episodes(cold_panel, monkeypatch):
    support_calls = Mock(wraps=api().supported)
    monkeypatch.setattr(api(), "supported", support_calls)
    results = []
    for share in (0.0, 1.0):
        rng = np.random.default_rng(19)
        recorded = Mock(wraps=rng)
        anchors = []

        def record_anchor(low, high):
            anchor = rng.integers(low, high)
            anchors.append(anchor)
            return anchor

        recorded.integers.side_effect = record_anchor
        support_calls.reset_mock()
        _, plans, _ = plan_cold(
            cold_panel,
            recorded,
            cold_share=share,
            episodes_per_panel=4,
            episodes_per_batch=4,
            path_share=0.5,
            drop=(("peers", 0.5), ("zeros", 0.5)),
        )
        (anchor,) = anchors  # Each run accepted its first choose.
        assert support_calls.call_args_list[0].args[2] == plans[0].item_id
        results.append((plans, anchor, rng.bit_generator.state))
    (ordinary, ordinary_anchor, ordinary_state), (cold, cold_anchor, cold_state) = results
    assert ordinary[0].item_id == cold[0].item_id
    assert ordinary_anchor == cold_anchor == ordinary[0].request.origins[-1]
    assert not ordinary[0].cold and cold[0].cold
    assert len(ordinary) == len(cold) == 4 and ordinary[1:] == cold[1:]
    assert ordinary_state == cold_state


@pytest.mark.parametrize("rejection", [None, "fallback", "no_labels"])
def test_cold_retry_and_rejection(cold_panel, monkeypatch, rejection):
    values = cold_panel["generated"].panel.items[1].targets["y0"]
    if rejection == "fallback":
        values[:4] = 1
    elif rejection == "no_labels":
        values[4:] = np.nan
    support = Mock(wraps=api().supported, side_effect=[False, DEFAULT])
    monkeypatch.setattr(api(), "supported", support)
    rng = scripted_cold_rng(indices=(0, 1), steps=(3, 4), coins=(0.6, 0.7))
    _, plans, counts = plan_cold(cold_panel, rng)
    assert support.call_count == 2
    assert [(c.args[2:4], c.kwargs["cutoff"]) for c in support.call_args_list] == [
        (("0", 35), 35),
        (("1", 36), 36),
    ]
    assert rng.mock_calls[5:14] == [
        call.spawn(1),
        call.choice([0, 1]),
        call.integers(48, 64),
        call.cold.random(),
        call.cold.integers(3, 5),
        call.choice([0, 1]),
        call.integers(48, 64),
        call.cold.random(),
        call.cold.integers(3, 5),
    ]
    assert counts["cold_attempted"] == 2 and counts["cold_planned"] == int(rejection is None)
    if rejection:
        assert not plans and counts[rejection] == 1
    else:
        (plan,) = plans
        assert plan.cold and plan.request.origins == (36,)


def test_cold_counts_are_reported_apart_from_skips():
    c = static_batch_config()
    _, default = api().make_batch(c, 3)
    assert default == _COLD_BASE_REPORT
    assert "cold" not in default
    cold = replace(c, sampler=replace(c.sampler, cold_share=1.0))
    _, report = api().make_batch(cold, 3)
    assert set(report["cold"]) == {"attempted", "planned"}
    assert report["cold"]["attempted"] >= report["cold"]["planned"] > 0
    assert report["skips"].keys().isdisjoint({"cold_attempted", "cold_planned"})


def test_empty_cold_batch_reports_counts_apart_from_skips():
    c = static_batch_config(cold_share=1.0, episodes_per_batch=1, episodes_per_panel=1, max_cells=1)
    with pytest.raises(RuntimeError, match="no episodes after panel limit") as error:
        api().make_batch(c, 3)
    message = str(error.value)
    skips_text, cold_marker, cold_text = message.split("skips=", 1)[1].partition("; cold=")
    skips = literal_eval(skips_text)
    assert skips.keys().isdisjoint({"cold_attempted", "cold_planned"})
    assert skips["oversized"] > 0
    assert cold_marker, message
    cold = literal_eval(cold_text)
    assert cold["attempted"] >= cold["planned"] > 0


def test_cold_share_refuses_missing_as_zero_source_mix_override():
    c = static_batch_config(cold_share=1.0, origins=1)
    sources = SourceMix((replace(c.sources[0], control="missing_as_zero"),))
    with pytest.raises(RuntimeError, match=r"missing_as_zero.*cold_share"):
        api().make_batch(c, 3, sources=sources)


@pytest.mark.parametrize(
    "mechanism", ["known_driver", "event_response", "label_lag", "static_number"]
)
def test_cold_applies_to_two_mechanisms_only(mechanism):
    c = static_batch_config(path_share=0.5)
    source = replace(c.sources[0], recipes=((mechanism,),))
    p, record = TypedSource(source).panel(np.random.default_rng(2))
    default_rng, cold_rng = (Mock(wraps=np.random.default_rng(5)) for _ in range(2))
    _, ordinary, _ = api().plan_panel(p, record, c.sampler, source, default_rng)
    _, cold, counts = api().plan_panel(
        p, record, replace(c.sampler, cold_share=1.0), source, cold_rng
    )
    assert ordinary and cold == ordinary and not any(plan.cold for plan in cold)
    assert counts["cold_attempted"] == counts["cold_planned"] == 0
    default_rng.spawn.assert_not_called()
    cold_rng.spawn.assert_not_called()
    assert default_rng.mock_calls == cold_rng.mock_calls


def test_no_value_at_or_after_the_origin_reaches_a_cold_episode(cold_panel):
    p, (plan,), _ = plan_cold(cold_panel, scripted_cold_rng())
    request = plan.request
    original = EpisodeBuilder(p).build(request)
    changed = deepcopy(p)
    for i, item in enumerate(changed.items):
        if i != request.item_index:
            offset = max(0, request.origins[0] - item.start)
            for values in (*item.targets.values(), *item.dynamic.values()):
                assert values[offset:].size > 0
                values[offset:] = -9999
    rebuilt = EpisodeBuilder(changed).build(request)
    assert plan.cold and (original.row_role == ROLE_PEER).any()
    for field in fields(original):
        np.testing.assert_equal(
            getattr(rebuilt, field.name), getattr(original, field.name), field.name
        )


@pytest.mark.parametrize("path", ["static_category", "multi_label"])
def test_true_and_shuffled_builds_of_a_cold_entry(path):
    (entry,) = t1.eval_set(path, episodes=1, max_panels=80)
    generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
    item = next(item for item in generated.panel.items if item.item_id == entry["item_id"])
    assert entry["origin"] - item.start in (3, 4) and entry["origin"] == entry["cutoff"]
    true, null = (
        t1._build(generated, entry, t1.t1_config(path, arm, seed=7)) for arm in ("true", "null")
    )
    assert true.context_target == null.context_target == entry["target"]
    for field in ("origin_patches", "labels", "label_mask", "loc", "scale"):
        np.testing.assert_array_equal(getattr(true, field), getattr(null, field), err_msg=field)
    assert (
        not np.array_equal(true.x[true.row_role == ROLE_PEER], null.x[null.row_role == ROLE_PEER])
        or true.attr_names != null.attr_names
        or not np.array_equal(true.attr_stats, null.attr_stats)
    )


def test_cold_batches_are_equal_for_workers():
    c = static_batch_config(cold_share=1.0)
    serial, parallel = (list(islice(api().loader(replace(c, workers=n), 3), 4)) for n in (0, 2))
    for (k, batch, report), (j, other_batch, other_report) in zip(serial, parallel, strict=True):
        assert k == j and report == other_report and report["cold"]["planned"] > 0
        equal_batches(batch, other_batch)


@pytest.mark.parametrize("path", ["static_category", "multi_label"])
@pytest.mark.parametrize("arm", ["true", "null"])
def test_static_path_batches_are_full(path, arm, monkeypatch):
    c = t1.t1_config(path, arm, seed=7)
    draw = Mock(wraps=api()._draw)
    monkeypatch.setattr(api(), "_draw", draw)
    for k in (0, 1, 2):
        draw.reset_mock()
        batch, report = api().make_batch(c, k)
        assert batch.x.shape[0] == report["episodes"] == c.sampler.episodes_per_batch
        assert c.sampler.episodes_per_batch <= draw.call_count <= 4 * c.sampler.episodes_per_batch
        assert 0 < report["paths"][path] <= report["cold"]["planned"] <= report["cold"]["attempted"]
        print(f"{path}/{arm} batch {k}: {draw.call_count} panels, cold={report['cold']}")


def test_requests_share_patch_anchor_and_fit_context():
    p = panel()
    c = config().sampler
    for seed in range(10):
        _, plans, _ = api().plan_panel(p, {}, c, config().sources[0], np.random.default_rng(seed))
        for plan in plans:
            r = plan.request
            assert r.origins == tuple(sorted(r.origins)) and len(r.origins) <= 2
            assert all((r.origins[-1] - o) % 16 == 0 for o in r.origins)
            assert r.origins[0] >= r.origins[-1] - r.context + 16
            assert r.cutoff <= r.origins[0]
            EpisodeBuilder(p).build(r)


def test_short_items_use_own_cutoff_long_items_share_panel_cutoff():
    p = panel(512, count=3)
    short = replace(p.items[2], start=220, targets={"y0": np.arange(40, dtype=np.float64)})
    p = replace(p, items=(*p.items[:2], short))
    c = config(contexts=(256,), origins=4, cutoff_at_origin=0, designate_target=True).sampler
    seen = False
    for seed in range(100):
        _, plans, _ = api().plan_panel(p, {}, c, config().sources[0], np.random.default_rng(seed))
        long_cutoffs = [q.request.cutoff for q in plans if q.item_id != short.item_id]
        assert len(set(long_cutoffs)) <= 1
        for q in plans:
            if q.item_id == short.item_id:
                seen = True
                assert q.request.cutoff == q.request.origins[0]
                assert not EpisodeBuilder(p).build(q.request).fallback.any()
    assert seen


@pytest.mark.parametrize(
    "mechanisms",
    [(), ("calendar", "intermittent", "level_shift"), ("planned_event",), ("static_category",)],
)
def test_empty_structure_and_path_panels(mechanisms):
    c = config(path_share=1, designate_target=True)
    c = replace(
        c, sources=(replace(c.sources[0], recipes=(mechanisms,), max_items=24, length=(128, 160)),)
    )
    built, report = api().make_batch(c, 0)
    assert report["episodes"] and built.label_mask.any()
    if mechanisms in (("planned_event",), ("static_category",)):
        assert report["paths"].get(mechanisms[0], 0) > 0
    p, record = TypedSource(c.sources[0]).panel(np.random.default_rng(2))
    _, plans, _ = api().plan_panel(p, record, c.sampler, c.sources[0], np.random.default_rng(3))
    for q in plans:
        if q.mechanism == "planned_event":
            item = p.items[q.request.item_index]
            col = record["generated"].inputs_of(q.mechanism)[0].name
            left = q.request.origins[-1] - item.start
            assert (item.dynamic[col][left : left + q.request.horizon] == 1).any()
        if q.mechanism == "static_category":
            assert q.item_id in record["generated"].short_items


def test_protection_consumes_identical_drop_draws(monkeypatch):
    c = config(
        path_share=1,
        designate_target=True,
        drop=(("features", 1.0), ("zeros", 0.7), ("value_stats", 1.0)),
    )
    source = replace(c.sources[0], recipes=(("known_driver",),), length=(128, 160))
    p, record = TypedSource(source).panel(np.random.default_rng(1))
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    _, protected, _ = api().plan_panel(p, record, c.sampler, source, a)
    monkeypatch.setitem(api().PROTECT, "known_driver", ())
    _, unprotected, _ = api().plan_panel(p, record, c.sampler, source, b)
    assert a.bit_generator.state == b.bit_generator.state
    assert any(q.mechanism for q in protected)
    assert all("features" not in q.request.drop for q in protected if q.mechanism)
    assert all("features" in q.request.drop for q in unprotected)


def test_no_support_and_tiny_necessary_items_are_safe():
    g = generate_typed_panel(PanelRecipe(("static_category",), n_items=48, seed=1))
    assert any(len(i.targets["target"]) < 17 for i in g.panel.items if i.item_id in g.short_items)
    c = config(path_share=1, designate_target=True)
    record = {"source": "typed", "generated": g, "recipe": g.recipe}
    _, plans, _ = api().plan_panel(
        g.panel, record, c.sampler, c.sources[0], np.random.default_rng(0)
    )
    assert plans
    hidden = deepcopy(g)
    for item in hidden.panel.items:
        for name in item.static:
            item.static[name] = None
    _, plans, skips = api().plan_panel(
        hidden.panel,
        {**record, "generated": hidden},
        c.sampler,
        c.sources[0],
        np.random.default_rng(0),
        path_quota=2,
        ordinary_quota=2,
    )
    assert skips["no_support"] == 1 and len(plans) == 2
    assert all(q.mechanism is None for q in plans)


def test_no_labels_skipped_before_build():
    p = panel(length=17, count=1)
    p.items[0].targets["y0"][16:] = np.nan
    _, plans, skips = api().plan_panel(
        p, {}, config().sampler, config().sources[0], np.random.default_rng(1)
    )
    assert not plans and skips["no_labels"] == 1


def test_fallback_checked_on_uncontrolled_panel():
    p = panel(length=17, count=1)
    p.items[0].targets["y0"][:16] = 1
    p.items[0].targets["y0"][0] = np.nan
    source = replace(config().sources[0], control="missing_as_zero")
    _, plans, skips = api().plan_panel(
        p, {}, config(origins=1).sampler, source, np.random.default_rng(1)
    )
    assert not plans and skips["fallback"] == 1


def test_missing_as_zero_admission_uses_uncontrolled_record(monkeypatch):
    p = panel(length=17, count=4)
    complete = deepcopy(p)
    p.items[0].targets["y0"][:16] = np.nan
    p.items[0].targets["y0"][0] = 7.0
    filled = deepcopy(p)
    filled.items[0].targets["y0"][1:16] = 0.0
    # At the only possible cutoff (16), one observation becomes {7, 0}.
    assert np.isfinite(p.items[0].targets["y0"][:16]).sum() == 1
    assert np.unique(filled.items[0].targets["y0"][:16]).tolist() == [0.0, 7.0]
    generated = replace(
        generate_typed_panel(PanelRecipe(("calendar",), n_items=4, length=(17, 17))),
        panel=p,
        complete=complete,
    )
    record = {"source": "typed", "generated": generated}
    c = config(
        designate_target=True,
        origins=1,
        cutoff_at_origin=1,
        path_share=0.5,
        drop=(),
        drop_always=("panel_row",),
    )
    built = []
    real_build = api().EpisodeBuilder.build

    def track_build(builder, request):
        episode = real_build(builder, request)
        built.append((builder.panel.items[request.item_index].item_id, request))
        return episode

    monkeypatch.setattr(api().EpisodeBuilder, "build", track_build)
    results = []
    for control, exposed in ((None, p), ("missing_as_zero", filled)):
        arm = replace(c, sources=(replace(c.sources[0], control=control),))
        # The exposed control differs from the original in record["generated"].
        _, plans, skips = api().plan_panel(
            exposed, record, arm.sampler, arm.sources[0], np.random.default_rng(1)
        )
        assert skips == {"fallback": 1}
        assert {plan.item_id for plan in plans} == {"1", "2", "3"}
        built.clear()
        batch, report = api().make_batch(
            arm, 0, sources=fixed_mix(arm, FixedSource(exposed, record))
        )
        assert report["episodes"] == len(built) == 4
        assert report["skips"]["fallback"] >= 1
        assert report["paths"] == {"calendar": 2}
        assert {item_id for item_id, _ in built[:3]} == {"1", "2", "3"}
        assert all(item_id != "0" for item_id, _ in built)
        results.append((plans, skips, tuple(built), batch, report))
    true, null = results
    assert true[:3] == null[:3]  # Kept items and requests include later panel draws.
    assert true[4] == null[4]  # Includes skips and paths after filling the batch.
    equal_batches(true[3], null[3])


def test_sampled_recipes_never_keep_unsupervised():
    # Real sampled recipes, with a small item cap; batches remain tiny.
    c = replace(config(episodes_per_batch=1), sources=(SourceConfig("typed", 1, max_items=2),))
    for k in range(50):
        _, report = api().make_batch(c, k)
        assert report["skips"].get("unsupervised", 0) == 0


def test_designated_target_siblings_drop_keeps_label_cells():
    c = config(designate_target=True, drop=())
    c = replace(c, sources=(replace(c.sources[0], recipes=(("multi_target",),)),))
    a, ra = api().make_batch(c, 0)
    b, rb = api().make_batch(replace(c, sampler=replace(c.sampler, drop_always=("siblings",))), 0)
    assert a.context_target == b.context_target
    assert torch.equal(a.origin_patches, b.origin_patches)
    for batch in (a, b):
        assert not batch.label_mask.masked_select(~batch.keep[:, None, :, None]).any()
    assert torch.equal(a.labels[:, :, :1], b.labels[:, :, :1])
    assert torch.equal(a.label_mask[:, :, :1], b.label_mask[:, :, :1])
    assert ra["supervised"] == rb["supervised"]


def test_shuffle_admission_requests_labels_and_masks_equal():
    c = config(designate_target=True, path_share=1, drop=())
    c = replace(
        c,
        sources=(
            replace(c.sources[0], recipes=(("event_response",),), max_items=24, length=(128, 160)),
        ),
    )
    null = replace(c, sources=(replace(c.sources[0], control="shuffle"),))
    a, ra = api().make_batch(c, 3)
    b, rb = api().make_batch(null, 3)
    for name in ("labels", "label_mask", "origin_patches", "keep", "loc", "scale"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.context_target == b.context_target
    assert ra["skips"] == rb["skips"] and ra["paths"] == rb["paths"]
    p, record = TypedSource(c.sources[0]).panel(np.random.default_rng(2))
    pn, rn = TypedSource(null.sources[0]).panel(np.random.default_rng(2))
    assert (
        api().plan_panel(p, record, c.sampler, c.sources[0], np.random.default_rng(5))[1:]
        == (api().plan_panel(pn, rn, c.sampler, null.sources[0], np.random.default_rng(5))[1:])
    )


@pytest.mark.parametrize(
    "control,mechanism",
    [
        ("shuffle", "event_response"),
        ("no_dates", "calendar"),
        ("missing_as_zero", "missing_vs_zero"),
        ("drop_always", "seasonal_lag"),
    ],
)
def test_control_kept_fallback_and_supervised_cells_equal_over_ten_batches(control, mechanism):
    c = config(path_share=0.5, designate_target=True, origins=1, drop=())
    c = replace(
        c,
        sources=(replace(c.sources[0], recipes=((mechanism,),), max_items=24, length=(128, 160)),),
    )
    if control == "drop_always":
        null = replace(
            c,
            sampler=replace(
                c.sampler,
                drop_always=("rolling", "lags", "zeros"),
                summary_off=("zero_share", "missing_share", "log_length"),
            ),
        )
    else:
        null = replace(c, sources=(replace(c.sources[0], control=control),))
    for k in range(10):
        true, true_report = api().make_batch(c, k)
        other, other_report = api().make_batch(null, k)
        assert true.context_target == other.context_target
        assert torch.equal(true.origin_patches, other.origin_patches)
        assert torch.equal(true.keep, other.keep)
        kept = true.keep & true.forecast_mask
        assert kept.any()
        assert torch.equal(true.fallback[kept], other.fallback[kept])
        counts = []
        for batch in (true, other):
            steps = torch.arange(batch.labels.shape[-1])[None, :] < batch.horizon[:, None]
            valid = batch.keep & batch.forecast_mask & ~batch.fallback
            mask = batch.label_mask & valid[:, None, :, None] & steps[:, None, None, :]
            counts.append(mask.sum(dim=(1, 2, 3)))
        assert torch.equal(*counts)
        assert int(counts[0].sum()) == true_report["supervised"] == other_report["supervised"]
        assert true_report["paths"] == other_report["paths"]
        assert true_report["skips"] == other_report["skips"]


def test_missing_as_zero_equals_prefilled_panel_and_preserves_true_units():
    p = panel(length=17, count=2)
    for item in p.items:
        item.targets["y0"][3:5] = np.nan
    c = config(designate_target=True, origins=1, drop=(), cutoff_at_origin=1)
    true, _ = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(p)))
    null_config = replace(c, sources=(replace(c.sources[0], control="missing_as_zero"),))
    null, _ = api().make_batch(null_config, 0, sources=fixed_mix(null_config, FixedSource(p)))
    filled = deepcopy(p)
    for item in filled.items:
        item.targets["y0"][3:5] = 0
    expected, _ = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(filled)))
    equal_batches(null, expected)
    assert torch.equal(null.label_mask, true.label_mask)
    original = true.labels * true.scale[:, None, :, None] + true.loc[:, None, :, None]
    restored = null.labels * null.scale[:, None, :, None] + null.loc[:, None, :, None]
    torch.testing.assert_close(restored[true.label_mask], original[true.label_mask])
    assert np.isnan(p.items[0].targets["y0"][3])


def test_missing_as_zero_fills_corpus_history(tmp_path):
    """The control belongs to the sampler, so it reaches a corpus panel too."""
    pytest.importorskip("pyarrow")
    values = np.arange(64, dtype=np.float64)
    values[3:5] = np.nan
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[values]])
    source = SourceConfig("corpus", 1, root=str(tmp_path), control="missing_as_zero")
    base = config(designate_target=True, origins=1, drop=(), cutoff_at_origin=1)
    c = replace(base, sources=(source,))
    p, record = SourceMix(c.sources).panel(np.random.default_rng(0))
    target = record["target"]
    assert np.isnan(p.items[0].targets[target][3:5]).all()
    planned = api().plan_panel(p, record, c.sampler, source, np.random.default_rng(0))
    exposed, plans = planned[0], planned[1]
    assert plans
    assert (exposed.items[0].targets[target][3:5] == 0.0).all()
    assert np.isnan(p.items[0].targets[target][3:5]).all()  # The source's row is untouched.
    true = replace(c, sources=(replace(source, control=None),))
    uncontrolled = api().plan_panel(
        p, record, true.sampler, true.sources[0], np.random.default_rng(0)
    )
    assert np.isnan(uncontrolled[0].items[0].targets[target][3:5]).all()
    assert [plan.request for plan in plans] == [plan.request for plan in uncontrolled[1]]
    _, report = api().make_batch(c, 0)
    assert set(report["sources"]) == {"corpus"}


def test_summary_off_changes_only_named_entries():
    c = config(designate_target=True, drop=())
    c = replace(c, sources=(replace(c.sources[0], recipes=(("static_category",),), max_items=12),))
    true, _ = api().make_batch(c, 0)
    null, _ = api().make_batch(
        replace(c, sampler=replace(c.sampler, summary_off=("zero_share",))), 0
    )
    j = SUMMARY_FEATURES.index("zero_share")
    for name in ("row_summary", "row_summary_flags", "cand_summary", "cand_flags", "attr_stats"):
        expected = getattr(true, name).clone()
        expected[..., j] = 0
        if name == "attr_stats":
            expected[..., j + N_SUMMARY] = 0
        assert torch.equal(expected, getattr(null, name)), name
    for f in fields(true):
        if f.name not in (
            "row_summary",
            "row_summary_flags",
            "cand_summary",
            "cand_flags",
            "attr_stats",
        ):
            a, b = getattr(true, f.name), getattr(null, f.name)
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b


def test_corpus_designated_target_siblings_and_bad_values(tmp_path):
    pytest.importorskip("pyarrow")
    targets = [np.arange(64, dtype=np.float64) + j for j in range(3)]
    bad = np.arange(64, dtype=np.float64)
    bad[3] = np.inf
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[bad], targets])
    c = replace(config(drop=()), sources=(SourceConfig("corpus", 1, root=str(tmp_path)),))
    seen_bad = False
    for k in range(5):
        b, report = api().make_batch(c, k)
        seen_bad |= report["skips"].get("bad_values", 0) > 0
        assert report["sources"] == {"corpus": 4}
        assert all(t in names for t, names in zip(b.context_target, b.row_names, strict=True))
        assert b.label_mask[:, :, 1:].any()
        p, record = SourceMix(c.sources).panel(np.random.default_rng(0))
        _, plans, _ = api().plan_panel(p, record, c.sampler, c.sources[0], np.random.default_rng(0))
        assert len(plans) == 1 and plans[0].request.target == record["target"]
    assert seen_bad


def shaped_episode(rows, patches, known=0):
    e = EpisodeBuilder(panel(length=64, count=1)).build(EpisodeRequest(0, (32,), 16, 16))
    e.x = np.zeros((rows, patches, 16, 39), np.float32)
    e.patch_valid = np.ones((rows, patches), bool)
    e.known_rows = np.zeros(known, np.int16)
    e.xh = np.zeros((1, known, 1, 16, 39), np.float32)
    return e


def test_padded_budget_includes_cross_shape_and_known_decoder_rows(monkeypatch):
    c = config(episodes_per_batch=3, episodes_per_panel=3, max_cells=90_000)
    assert api()._cells([shaped_episode(1, 1, known=16)]) == 16 + 17 * 16
    episodes = iter([shaped_episode(40, 16), shaped_episode(1, 128)])
    monkeypatch.setattr(api().EpisodeBuilder, "build", lambda self, r: next(episodes))
    _, report = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(panel())))
    assert report["episodes"] == 1 and report["skips"]["budget"] == 1
    assert report["cells"] == 40 * 16 * 16 + 16


def test_oversized_skipped_then_stream_continues(monkeypatch):
    c = config(episodes_per_batch=1, max_cells=1000)
    episodes = iter([shaped_episode(40, 16), shaped_episode(1, 1)])
    monkeypatch.setattr(api().EpisodeBuilder, "build", lambda self, r: next(episodes))
    _, report = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(panel())))
    assert report["skips"]["oversized"] == 1 and report["episodes"] == 1


@pytest.mark.parametrize("stage", ["draw", "plan", "build"])
@pytest.mark.parametrize("error", [KeyError("injected"), ValueError("injected")])
def test_unexpected_exceptions_name_batch(monkeypatch, stage, error):
    fixed = FixedSource(panel())

    def fail(*args, **kwargs):
        raise error

    if stage == "draw":
        monkeypatch.setattr(fixed, "panel", fail)
    elif stage == "plan":
        monkeypatch.setattr(api(), "_plan_panel", fail)
    else:
        monkeypatch.setattr(api().EpisodeBuilder, "build", fail)
    with pytest.raises(RuntimeError, match="batch 17.*injected"):
        api().make_batch(config(), 17, sources=fixed_mix(config(), fixed))


def test_all_short_panels_fail_with_index_and_count():
    with pytest.raises(RuntimeError, match="batch 19.*no_room.*16"):
        api().make_batch(config(), 19, sources=fixed_mix(config(), FixedSource(panel(16))))


@pytest.mark.parametrize(
    "error,reason", [(ShortRow(), "short_row"), (CorpusValueError(), "bad_values")]
)
def test_only_sampler_skip_is_counted(monkeypatch, error, reason):
    fixed = FixedSource(panel())
    original = fixed.panel
    calls = 0

    def draw(rng):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return original(rng)

    monkeypatch.setattr(fixed, "panel", draw)
    _, report = api().make_batch(config(), 0, sources=fixed_mix(config(), fixed))
    assert report["skips"][reason] == 1


def test_unlabelled_episode_is_never_built(monkeypatch):
    p = panel(length=17, count=1)
    p.items[0].targets["y0"][16] = np.nan

    def forbidden(*args):
        pytest.fail("an unlabelled request was built")

    monkeypatch.setattr(api().EpisodeBuilder, "build", forbidden)
    with pytest.raises(RuntimeError, match="no_labels.*16"):
        api().make_batch(config(), 0, sources=fixed_mix(config(), FixedSource(p)))


@pytest.mark.parametrize("stage", ["plan", "build"])
def test_sampler_skip_at_planning_and_building_is_counted(monkeypatch, stage):
    name = "_plan_panel" if stage == "plan" else "build"
    owner = api() if stage == "plan" else api().EpisodeBuilder
    original = getattr(owner, name)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ShortRow()
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail_once)
    _, report = api().make_batch(config(), 0, sources=fixed_mix(config(), FixedSource(panel())))
    assert report["skips"]["short_row"] == 1 and report["episodes"] == 4
