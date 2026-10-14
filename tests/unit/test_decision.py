"""
tests/unit/test_decision.py
----------------------------
Unit tests for the Decision Layer:
  - DecisionRecord creation and validation
  - DecisionLog append, filter, serialisation
  - Severity ordering
"""

from __future__ import annotations

import json

import pytest

from mentorml.core.decision import DecisionLog, DecisionRecord, Severity


class TestSeverity:
    """Tests for the Severity enum."""

    def test_severity_values(self) -> None:
        assert Severity.INFO.value == "INFO"
        assert Severity.WARNING.value == "WARNING"
        assert Severity.CRITICAL.value == "CRITICAL"

    def test_severity_is_string(self) -> None:
        """Severity should serialise to string without a custom encoder."""
        assert isinstance(Severity.INFO, str)
        assert json.dumps({"s": Severity.WARNING}) == '{"s": "WARNING"}'


class TestDecisionRecord:
    """Tests for DecisionRecord construction and validation."""

    def test_basic_construction(self, sample_record: DecisionRecord) -> None:
        assert sample_record.component == "TestComponent"
        assert sample_record.action == "test_action"
        assert sample_record.severity == Severity.INFO
        assert sample_record.data == {"key": "value", "count": 42}

    def test_timestamp_is_set_automatically(
        self, sample_record: DecisionRecord
    ) -> None:
        """Timestamp should be populated even if not explicitly provided."""
        assert sample_record.timestamp is not None

    def test_empty_component_raises(self) -> None:
        with pytest.raises(ValueError, match="component"):
            DecisionRecord(
                component="",
                action="some_action",
                rationale="some rationale",
                data={},
            )

    def test_empty_action_raises(self) -> None:
        with pytest.raises(ValueError, match="action"):
            DecisionRecord(
                component="Comp",
                action="",
                rationale="some rationale",
                data={},
            )

    def test_empty_rationale_raises(self) -> None:
        with pytest.raises(ValueError, match="rationale"):
            DecisionRecord(
                component="Comp",
                action="act",
                rationale="",
                data={},
            )

    def test_string_severity_normalised(self) -> None:
        """Passing a raw string for severity should be normalised to Severity."""
        record = DecisionRecord(
            component="Comp",
            action="act",
            rationale="reason",
            data={},
            severity="warning",  # type: ignore[arg-type]
        )
        assert record.severity == Severity.WARNING

    def test_to_dict_is_json_serialisable(
        self, sample_record: DecisionRecord
    ) -> None:
        d = sample_record.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert "TestComponent" in serialised
        assert "INFO" in serialised

    def test_str_representation(self, sample_record: DecisionRecord) -> None:
        s = str(sample_record)
        assert "INFO" in s
        assert "TestComponent" in s
        assert "test_action" in s


class TestDecisionLog:
    """Tests for DecisionLog behaviour."""

    def test_empty_log_has_zero_length(self, empty_log: DecisionLog) -> None:
        assert len(empty_log) == 0

    def test_append_increases_length(
        self, empty_log: DecisionLog, sample_record: DecisionRecord
    ) -> None:
        empty_log.append(sample_record)
        assert len(empty_log) == 1

    def test_extend_appends_multiple(
        self, empty_log: DecisionLog, sample_record: DecisionRecord
    ) -> None:
        empty_log.extend([sample_record, sample_record])
        assert len(empty_log) == 2

    def test_iter_yields_records(self, populated_log: DecisionLog) -> None:
        records = list(populated_log)
        assert len(records) == 3
        assert all(isinstance(r, DecisionRecord) for r in records)

    def test_getitem(self, populated_log: DecisionLog) -> None:
        assert populated_log[0].action == "test_action"

    def test_filter_info_returns_all(self, populated_log: DecisionLog) -> None:
        """Filtering at INFO level should return all 3 records."""
        assert len(populated_log.filter(Severity.INFO)) == 3

    def test_filter_warning_excludes_info(self, populated_log: DecisionLog) -> None:
        """Filtering at WARNING should return WARNING + CRITICAL (2 records)."""
        results = populated_log.filter(Severity.WARNING)
        assert len(results) == 2
        assert all(r.severity != Severity.INFO for r in results)

    def test_filter_critical_returns_one(self, populated_log: DecisionLog) -> None:
        results = populated_log.filter(Severity.CRITICAL)
        assert len(results) == 1
        assert results[0].severity == Severity.CRITICAL

    def test_filter_by_component(self, populated_log: DecisionLog) -> None:
        results = populated_log.filter_by_component("OtherComponent")
        assert len(results) == 1
        assert results[0].component == "OtherComponent"

    def test_to_json_is_valid(self, populated_log: DecisionLog) -> None:
        raw = populated_log.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_repr_shows_counts(self, populated_log: DecisionLog) -> None:
        r = repr(populated_log)
        assert "total=3" in r
        assert "INFO=1" in r
        assert "WARNING=1" in r
        assert "CRITICAL=1" in r