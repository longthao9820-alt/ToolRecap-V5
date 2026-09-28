from __future__ import annotations

import copy

import pytest

from catalog_test_helpers import populated_store
from toolrecap_v4.analysis.finalizer.catalog import CatalogBuilder, canonical_catalog_bytes
from toolrecap_v4.analysis.finalizer.packing import PACKING_VERSION, pack_catalog, packed_catalog_bytes, unpack_catalog
from toolrecap_v4.errors import CatalogPackingError


@pytest.fixture
def catalog(tmp_path):
    store, revision, episodes, _ = populated_store(tmp_path)
    return CatalogBuilder().build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )


def test_pack_unpack_is_exact_and_byte_stable(catalog):
    first = pack_catalog(catalog)
    second = pack_catalog(catalog)
    assert first == second
    assert packed_catalog_bytes(first) == packed_catalog_bytes(second)
    assert unpack_catalog(first) == catalog
    assert first["packing_version"] == PACKING_VERSION


def test_packing_preserves_every_semantic_field_without_truncation(catalog):
    unpacked = unpack_catalog(pack_catalog(catalog))
    assert unpacked.to_dict() == catalog.to_dict()
    assert [item.evidence_id for item in unpacked.items] == [item.evidence_id for item in catalog.items]
    assert [episode.episode_id for episode in unpacked.ordered_episodes] == ["E01", "E02"]
    assert unpacked.items[0].factual_observation == "Beth says “héllo” — こんにちは."
    assert unpacked.items[0].uncertainty == ("speaker label absent",)
    assert unpacked.items[0].entities == ("Beth", "Jamie")
    assert unpacked.items[2].factual_observation == unpacked.items[3].factual_observation
    assert unpacked.items[2].evidence_id != unpacked.items[3].evidence_id
    assert len(unpacked.items[2].factual_observation) > 500
    assert unpacked.ordered_episodes[1].evidence_count == 0
    assert len(canonical_catalog_bytes(unpacked)) == len(canonical_catalog_bytes(catalog))


def _mutate(packed, action):
    value = copy.deepcopy(packed)
    action(value)
    return value


@pytest.mark.parametrize("action", [
    lambda p: p.update(packing_version="invalid"),
    lambda p: p["items"][0].__setitem__(6, 999999),
    lambda p: p["episodes"][0].__setitem__(1, 999999),
    lambda p: p["items"][0].__setitem__(1, 999999),
    lambda p: p["items"][0].pop(),
    lambda p: p["items"][0].__setitem__(3, 1.5),
    lambda p: p["items"].append(copy.deepcopy(p["items"][0])),
    lambda p: p.update(item_count=p["item_count"] + 1),
    lambda p: p["completeness"].__setitem__(4, 0),
    lambda p: p.update(catalog_hash="bad"),
])
def test_corrupt_packed_catalog_is_rejected(catalog, action):
    packed = pack_catalog(catalog)
    with pytest.raises(CatalogPackingError):
        unpack_catalog(_mutate(packed, action))
