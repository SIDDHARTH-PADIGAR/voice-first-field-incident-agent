"""Authorized, read-only asset lookup."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import re

from field_incident_agent.models import Asset


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} is not a valid identifier.")


@dataclass(frozen=True, slots=True)
class AccessContext:
    """Trusted authorization context supplied by the backend."""

    principal_id: str
    authorized_site_ids: frozenset[str]

    def __post_init__(self) -> None:
        _validate_identifier(self.principal_id, "principal_id")

        if not isinstance(self.authorized_site_ids, frozenset):
            raise ValueError("authorized_site_ids must be a frozenset.")

        for site_id in self.authorized_site_ids:
            _validate_identifier(site_id, "authorized site ID")


class LookupStatus(StrEnum):
    NOT_FOUND = "not_found"
    FOUND = "found"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class AssetLookupResult:
    status: LookupStatus
    matches: tuple[Asset, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, LookupStatus):
            raise ValueError("status must be a LookupStatus.")

        if not isinstance(self.matches, tuple):
            raise ValueError("matches must be a tuple.")

        expected_status = (
            LookupStatus.NOT_FOUND
            if len(self.matches) == 0
            else LookupStatus.FOUND
            if len(self.matches) == 1
            else LookupStatus.AMBIGUOUS
        )

        if self.status is not expected_status:
            raise ValueError("Lookup status does not match the results.")

        if not all(isinstance(asset, Asset) for asset in self.matches):
            raise ValueError("Every match must be an Asset.")


class AssetRegistry:
    """Read-only registry restricted by trusted site authorization."""

    def __init__(self, assets: Iterable[Asset]) -> None:
        records = tuple(assets)

        if not all(isinstance(asset, Asset) for asset in records):
            raise ValueError("Registry entries must be Asset instances.")

        keys = [(asset.site_id, asset.asset_id) for asset in records]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate asset ID within the same site.")

        self._assets = tuple(
            sorted(records, key=lambda asset: (asset.site_id, asset.asset_id))
        )

    @staticmethod
    def _result(matches: tuple[Asset, ...]) -> AssetLookupResult:
        if len(matches) == 0:
            status = LookupStatus.NOT_FOUND
        elif len(matches) == 1:
            status = LookupStatus.FOUND
        else:
            status = LookupStatus.AMBIGUOUS

        return AssetLookupResult(status=status, matches=matches)

    @staticmethod
    def _validate_access(access: AccessContext) -> None:
        if not isinstance(access, AccessContext):
            raise TypeError("access must be a trusted AccessContext.")

    def lookup_by_id(
        self,
        asset_id: str,
        access: AccessContext,
    ) -> AssetLookupResult:
        """Look up an exact ID without revealing unauthorized records."""
        self._validate_access(access)
        _validate_identifier(asset_id, "asset_id")

        matches = tuple(
            asset
            for asset in self._assets
            if asset.asset_id == asset_id
            and asset.site_id in access.authorized_site_ids
        )

        return self._result(matches)

    def search_by_clue(
        self,
        site_id: str,
        clue: str,
        access: AccessContext,
    ) -> AssetLookupResult:
        """Search one authorized site by ID, display name, or asset type."""
        self._validate_access(access)
        _validate_identifier(site_id, "site_id")

        if site_id not in access.authorized_site_ids:
            raise PermissionError("Caller is not authorized for this site.")

        if not isinstance(clue, str) or not clue.strip():
            raise ValueError("Search clue cannot be empty.")

        normalized_clue = clue.strip().casefold()

        if len(normalized_clue) > 300:
            raise ValueError("Search clue cannot exceed 300 characters.")

        matches = tuple(
            asset
            for asset in self._assets
            if asset.site_id == site_id
            and normalized_clue
            in f"{asset.asset_id} {asset.display_name} {asset.asset_type}".casefold()
        )

        return self._result(matches)