"""
mentorml.core.decision
----------------------
The Decision Layer — the central nervous system of mentorml.

Every meaningful action taken by any mentorml component must emit a
``DecisionRecord``.  A ``DecisionLog`` accumulates these records and acts
as the single source of truth for report generation, business narration,
and audit trails.

Design rationale
~~~~~~~~~~~~~~~~
Rather than scattering ``print()`` calls or log messages through the
codebase, we treat *every decision* as structured data.  This enables:

- **Reproducibility**: the full reasoning chain can be replayed or
  serialised to JSON.
- **Report generation**: the HTML report is built directly from the log.
- **Business narration**: the ``Narrator`` component reads the log and
  translates technical decisions into plain-English insights.
- **Auditing**: regulators and clients can inspect what the system did
  and why.

Usage
~~~~~
::

    from mentorml.core.decision import DecisionLog, DecisionRecord, Severity

    log = DecisionLog()

    record = DecisionRecord(
        component="DataQualityAssessor",
        action="drop_column",
        rationale=(
            "Column 'user_id' has 0% variance — it carries no predictive "
            "signal and will be dropped."
        ),
        data={"column": "user_id", "unique_values": 1},
        severity=Severity.WARNING,
    )
    log.append(record)

    # Serialise to JSON for persistence / report generation
    print(log.to_json())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """
    Categorises the impact level of a decision.

    Using ``str`` as a mixin means ``Severity.WARNING`` serialises to
    ``"WARNING"`` in JSON without a custom encoder.

    Attributes
    ----------
    INFO :
        Routine observation — no action required.  Example: "Dataset has
        1 200 rows and 18 features."
    WARNING :
        Something noteworthy that the user should be aware of.  Example:
        "Column 'age' has 12% missing values — imputing with median."
    CRITICAL :
        A significant issue that may materially affect model quality.
        Example: "Target column has severe class imbalance (98 / 2 split)."
    """

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# DecisionRecord
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """
    An immutable, structured record of a single decision made by mentorml.

    Parameters
    ----------
    component : str
        The name of the mentorml component that made this decision.
        Convention: use the class name, e.g. ``"DataQualityAssessor"``.
    action : str
        A short, machine-readable identifier for the action taken.
        Convention: ``"snake_case_verb_noun"``, e.g. ``"impute_missing"``.
    rationale : str
        A full human-readable explanation of *why* this action was taken.
        This is the text that will appear in reports and business narratives.
        Write it for a non-technical stakeholder.
    data : dict[str, Any]
        Structured metadata about the decision.  Include enough detail that
        the decision can be reconstructed from this dict alone.
    severity : Severity
        Impact level of this decision.  Defaults to ``Severity.INFO``.
    timestamp : datetime | None
        UTC timestamp of when the record was created.  Automatically set
        to ``datetime.now(tz=timezone.utc)`` if not provided.

    Examples
    --------
    ::

        record = DecisionRecord(
            component="ExplainablePreprocessor",
            action="encode_categorical",
            rationale=(
                "Column 'city' has 23 unique values — using target encoding "
                "instead of one-hot to avoid high-dimensional sparse features."
            ),
            data={"column": "city", "strategy": "target_encoding", "n_unique": 23},
            severity=Severity.INFO,
        )
    """

    component: str
    action: str
    rationale: str
    data: dict[str, Any]
    severity: Severity = Severity.INFO
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def __post_init__(self) -> None:
        """Validate fields immediately after construction."""
        if not self.component:
            raise ValueError("DecisionRecord.component must be a non-empty string.")
        if not self.action:
            raise ValueError("DecisionRecord.action must be a non-empty string.")
        if not self.rationale:
            raise ValueError("DecisionRecord.rationale must be a non-empty string.")
        # Normalise severity — allows passing raw strings like "WARNING"
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity.upper())

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the record to a plain Python dict.

        The ``timestamp`` is converted to an ISO-8601 string so the dict
        is directly JSON-serialisable.

        Returns
        -------
        dict[str, Any]
            A JSON-friendly representation of this record.
        """
        return {
            "component": self.component,
            "action": self.action,
            "rationale": self.rationale,
            "data": self.data,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.component}.{self.action} — "
            f"{self.rationale}"
        )


# ---------------------------------------------------------------------------
# DecisionLog
# ---------------------------------------------------------------------------


class DecisionLog:
    """
    An append-only, ordered collection of ``DecisionRecord`` objects.

    The log is the single source of truth for what mentorml did during a
    session.  It is passed through the pipeline and accumulated by each
    component.

    It supports iteration, filtering by severity, and full JSON
    serialisation.

    Examples
    --------
    ::

        log = DecisionLog()
        log.append(record)

        for r in log.filter(Severity.WARNING):
            print(r)

        with open("decisions.json", "w") as f:
            f.write(log.to_json())
    """

    def __init__(self) -> None:
        self._records: list[DecisionRecord] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, record: DecisionRecord) -> None:
        """
        Append a single ``DecisionRecord`` to the log.

        Also mirrors the record to the Python logging system at the
        appropriate level so that standard log handlers (file, stream, etc.)
        receive everything automatically.

        Parameters
        ----------
        record : DecisionRecord
            The record to append.
        """
        self._records.append(record)
        _level_map = {
            Severity.INFO: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.CRITICAL: logging.CRITICAL,
        }
        logger.log(_level_map[record.severity], str(record))

    def extend(self, records: list[DecisionRecord]) -> None:
        """
        Append multiple records at once.

        Parameters
        ----------
        records : list[DecisionRecord]
            Records to append, in order.
        """
        for record in records:
            self.append(record)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def filter(self, severity: Severity) -> list[DecisionRecord]:
        """
        Return all records at or above the given severity level.

        Severity ordering: INFO < WARNING < CRITICAL.

        Parameters
        ----------
        severity : Severity
            Minimum severity level to include.

        Returns
        -------
        list[DecisionRecord]
            Filtered records in chronological order.
        """
        _order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
        threshold = _order[severity]
        return [r for r in self._records if _order[r.severity] >= threshold]

    def filter_by_component(self, component: str) -> list[DecisionRecord]:
        """
        Return all records emitted by a specific component.

        Parameters
        ----------
        component : str
            Exact component name to match.

        Returns
        -------
        list[DecisionRecord]
            Filtered records in chronological order.
        """
        return [r for r in self._records if r.component == component]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict[str, Any]]:
        """
        Serialise the full log to a list of plain dicts.

        Returns
        -------
        list[dict[str, Any]]
            JSON-serialisable representation of all records.
        """
        return [r.to_dict() for r in self._records]

    def to_json(self, indent: int = 2) -> str:
        """
        Serialise the full log to a JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level.  Defaults to 2.

        Returns
        -------
        str
            Pretty-printed JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    # Python protocol support
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[DecisionRecord]:
        return iter(self._records)

    def __getitem__(self, index: int) -> DecisionRecord:
        return self._records[index]

    def __repr__(self) -> str:
        counts = {s: 0 for s in Severity}
        for r in self._records:
            counts[r.severity] += 1
        return (
            f"DecisionLog("
            f"total={len(self._records)}, "
            f"INFO={counts[Severity.INFO]}, "
            f"WARNING={counts[Severity.WARNING]}, "
            f"CRITICAL={counts[Severity.CRITICAL]})"
        )