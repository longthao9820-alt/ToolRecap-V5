"""Semantic Versioning (SemVer 2.0.0) parser and comparator."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional, Tuple, Union

SEMVER_REGEX = re.compile(
    r"^[vV]?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

Identifier = Union[int, str]


@dataclass(frozen=True)
class SemVer:
    """SemVer 2.0.0 data structure supporting full precedence comparison."""

    major: int
    minor: int
    patch: int
    prerelease: Tuple[Identifier, ...] = ()
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_str: str) -> SemVer:
        """Parse semver string into SemVer instance.
        
        Supports optional leading 'v' and conforms to SemVer 2.0.0 specification.
        Raises ValueError if format is invalid.
        """
        if not isinstance(version_str, str):
            raise ValueError(f"Version must be a string, got {type(version_str).__name__}")

        version_str = version_str.strip()
        match = SEMVER_REGEX.match(version_str)
        if not match:
            raise ValueError(f"Invalid SemVer string: {version_str!r}")

        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))

        raw_prerelease = match.group(4)
        prerelease_parts: Tuple[Identifier, ...] = ()
        if raw_prerelease:
            parts = []
            for p in raw_prerelease.split("."):
                if p.isdigit():
                    parts.append(int(p))
                else:
                    parts.append(p)
            prerelease_parts = tuple(parts)

        build = match.group(5)
        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease_parts,
            build=build,
        )

    def _compare_prerelease(self, other_pre: Tuple[Identifier, ...]) -> int:
        """Compare prerelease tuples per SemVer 2.0.0 rules.
        
        Returns:
            -1 if self < other
             0 if self == other
             1 if self > other
        """
        # A normal version has greater precedence than a pre-release version
        if not self.prerelease and other_pre:
            return 1
        if self.prerelease and not other_pre:
            return -1
        if not self.prerelease and not other_pre:
            return 0

        for a, b in zip(self.prerelease, other_pre):
            if a == b:
                continue
            # Numeric identifiers always have lower precedence than non-numeric
            if isinstance(a, int) and isinstance(b, str):
                return -1
            if isinstance(a, str) and isinstance(b, int):
                return 1
            if isinstance(a, int) and isinstance(b, int):
                return -1 if a < b else 1
            if isinstance(a, str) and isinstance(b, str):
                return -1 if a < b else 1

        # A larger set of pre-release fields has a higher precedence than a smaller set
        if len(self.prerelease) < len(other_pre):
            return -1
        if len(self.prerelease) > len(other_pre):
            return 1
        return 0

    def _cmp(self, other: Any) -> int:
        if not isinstance(other, SemVer):
            return NotImplemented

        if (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch):
            return -1
        if (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch):
            return 1

        return self._compare_prerelease(other.prerelease)

    def __lt__(self, other: Any) -> bool:
        res = self._cmp(other)
        if res is NotImplemented:
            return NotImplemented
        return res < 0

    def __le__(self, other: Any) -> bool:
        res = self._cmp(other)
        if res is NotImplemented:
            return NotImplemented
        return res <= 0

    def __eq__(self, other: Any) -> bool:
        res = self._cmp(other)
        if res is NotImplemented:
            return NotImplemented
        return res == 0

    def __ge__(self, other: Any) -> bool:
        res = self._cmp(other)
        if res is NotImplemented:
            return NotImplemented
        return res >= 0

    def __gt__(self, other: Any) -> bool:
        res = self._cmp(other)
        if res is NotImplemented:
            return NotImplemented
        return res > 0

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += f"-{'.'.join(str(p) for p in self.prerelease)}"
        if self.build:
            s += f"+{self.build}"
        return s
