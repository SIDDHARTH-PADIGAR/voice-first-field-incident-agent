import pytest

from field_incident_agent.models import Asset
from field_incident_agent.registry import (
    AccessContext,
    AssetLookupResult,
    AssetRegistry,
    LookupStatus,
)


@pytest.fixture
def registry():
    return AssetRegistry([
        Asset(
            asset_id="PUMP-001",
            site_id="SITE-001",
            asset_type="pump",
            display_name="Main water pump",
        ),
        Asset(
            asset_id="PUMP-002",
            site_id="SITE-001",
            asset_type="pump",
            display_name="Backup water pump",
        ),
        Asset(
            asset_id="FAN-001",
            site_id="SITE-002",
            asset_type="fan",
            display_name="Server room exhaust fan",
        ),
    ])


@pytest.fixture
def site_one_access():
    return AccessContext(
        principal_id="TECH-001",
        authorized_site_ids=frozenset({"SITE-001"}),
    )


def test_exact_lookup_finds_authorized_asset(registry, site_one_access):
    result = registry.lookup_by_id("PUMP-001", site_one_access)

    assert result.status is LookupStatus.FOUND
    assert result.matches[0].display_name == "Main water pump"


def test_unknown_asset_returns_not_found(registry, site_one_access):
    result = registry.lookup_by_id("PUMP-999", site_one_access)

    assert result.status is LookupStatus.NOT_FOUND
    assert result.matches == ()


def test_unauthorized_asset_is_not_revealed(registry, site_one_access):
    result = registry.lookup_by_id("FAN-001", site_one_access)

    assert result.status is LookupStatus.NOT_FOUND
    assert result.matches == ()


def test_clue_search_reports_multiple_matches(registry, site_one_access):
    result = registry.search_by_clue(
        "SITE-001",
        "water pump",
        site_one_access,
    )

    assert result.status is LookupStatus.AMBIGUOUS
    assert len(result.matches) == 2
    assert {asset.asset_id for asset in result.matches} == {
        "PUMP-001",
        "PUMP-002",
    }


def test_clue_search_is_limited_to_requested_site(
    registry,
    site_one_access,
):
    result = registry.search_by_clue(
        "SITE-001",
        "pump",
        site_one_access,
    )

    assert all(asset.site_id == "SITE-001" for asset in result.matches)


def test_search_for_unauthorized_site_is_rejected(
    registry,
    site_one_access,
):
    with pytest.raises(PermissionError, match="not authorized"):
        registry.search_by_clue(
            "SITE-002",
            "fan",
            site_one_access,
        )


def test_empty_search_clue_is_rejected(registry, site_one_access):
    with pytest.raises(ValueError, match="cannot be empty"):
        registry.search_by_clue(
            "SITE-001",
            "   ",
            site_one_access,
        )


def test_duplicate_asset_id_within_site_is_rejected():
    asset = Asset(
        asset_id="PUMP-001",
        site_id="SITE-001",
        asset_type="pump",
        display_name="Main water pump",
    )

    with pytest.raises(ValueError, match="Duplicate asset ID"):
        AssetRegistry([asset, asset])


def test_authorization_sites_must_be_immutable():
    with pytest.raises(ValueError, match="frozenset"):
        AccessContext(
            principal_id="TECH-001",
            authorized_site_ids={"SITE-001"},
        )


def test_lookup_result_rejects_inconsistent_status():
    with pytest.raises(ValueError, match="does not match"):
        AssetLookupResult(
            status=LookupStatus.FOUND,
            matches=(),
        )