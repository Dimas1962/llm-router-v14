"""
STATE-SNAPSHOT System - Component 10
Complete system state capture and restoration
"""

import json
import pickle
import gzip
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from copy import deepcopy


@dataclass
class ComponentState:
    """State of a single component"""
    component_name: str
    stats: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """Complete system state snapshot"""
    version: str
    timestamp: datetime
    components: Dict[str, ComponentState]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary"""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "components": {
                name: {
                    "component_name": state.component_name,
                    "stats": state.stats,
                    "config": state.config,
                    "metadata": state.metadata
                }
                for name, state in self.components.items()
            },
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemSnapshot":
        """Create snapshot from dictionary"""
        components = {}
        for name, comp_data in data["components"].items():
            components[name] = ComponentState(
                component_name=comp_data["component_name"],
                stats=comp_data["stats"],
                config=comp_data["config"],
                metadata=comp_data.get("metadata", {})
            )

        return cls(
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            components=components,
            metadata=data.get("metadata", {})
        )


@dataclass
class SnapshotDiff:
    """Difference between two snapshots"""
    from_version: str
    to_version: str
    added_components: List[str]
    removed_components: List[str]
    modified_components: Dict[str, Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


class StateSnapshotSystem:
    """
    STATE-SNAPSHOT System

    Features:
    - Capture complete system state (all 9 components)
    - Snapshot serialization (JSON/pickle)
    - State restoration from snapshots
    - Diff calculation between snapshots
    - Rollback to previous states
    - Snapshot versioning
    - Compression for large snapshots
    """

    def __init__(self):
        """Initialize STATE-SNAPSHOT System"""
        self.current_version = "1.0.0"
        self.snapshots: List[SystemSnapshot] = []
        self.max_snapshots = 10  # Keep last 10 snapshots

        # Statistics
        self.stats = {
            "total_snapshots": 0,
            "total_restorations": 0,
            "total_diffs": 0,
            "total_rollbacks": 0
        }

    def capture_snapshot(
        self,
        components: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemSnapshot:
        """
        Capture system snapshot

        Args:
            components: Dictionary of components to capture
            metadata: Optional metadata

        Returns:
            SystemSnapshot
        """
        component_states = {}

        for name, component in components.items():
            # Extract stats (most components have get_stats() method)
            stats = {}
            if hasattr(component, 'get_stats'):
                stats = component.get_stats()
            elif hasattr(component, 'stats'):
                stats = getattr(component, 'stats', {})

            # Extract config
            config = {}
            for attr in dir(component):
                if not attr.startswith('_') and not callable(getattr(component, attr)):
                    value = getattr(component, attr)
                    # Only capture simple types
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        if attr not in ['stats', 'cached_context']:  # Skip these
                            config[attr] = value

            component_states[name] = ComponentState(
                component_name=name,
                stats=stats,
                config=config
            )

        snapshot = SystemSnapshot(
            version=self.current_version,
            timestamp=datetime.now(),
            components=component_states,
            metadata=metadata or {}
        )

        # Add to history
        self.snapshots.append(snapshot)

        # Limit history size
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        # Update stats
        self.stats["total_snapshots"] += 1

        return snapshot

    def serialize_json(self, snapshot: SystemSnapshot) -> str:
        """
        Serialize snapshot to JSON

        Args:
            snapshot: Snapshot to serialize

        Returns:
            JSON string
        """
        return json.dumps(snapshot.to_dict(), indent=2)

    def deserialize_json(self, json_str: str) -> SystemSnapshot:
        """
        Deserialize snapshot from JSON

        Args:
            json_str: JSON string

        Returns:
            SystemSnapshot
        """
        data = json.loads(json_str)
        return SystemSnapshot.from_dict(data)

    def serialize_pickle(self, snapshot: SystemSnapshot) -> bytes:
        """
        Serialize snapshot to pickle

        Args:
            snapshot: Snapshot to serialize

        Returns:
            Pickled bytes
        """
        return pickle.dumps(snapshot)

    def deserialize_pickle(self, data: bytes) -> SystemSnapshot:
        """
        Deserialize snapshot from pickle

        Args:
            data: Pickled bytes

        Returns:
            SystemSnapshot
        """
        return pickle.loads(data)

    def compress_snapshot(self, snapshot: SystemSnapshot) -> bytes:
        """
        Compress snapshot

        Args:
            snapshot: Snapshot to compress

        Returns:
            Compressed bytes
        """
        # Serialize to JSON first
        json_data = self.serialize_json(snapshot).encode('utf-8')

        # Compress with gzip
        compressed = gzip.compress(json_data, compresslevel=9)

        return compressed

    def decompress_snapshot(self, compressed_data: bytes) -> SystemSnapshot:
        """
        Decompress snapshot

        Args:
            compressed_data: Compressed bytes

        Returns:
            SystemSnapshot
        """
        # Decompress
        json_data = gzip.decompress(compressed_data).decode('utf-8')

        # Deserialize
        return self.deserialize_json(json_data)

    def calculate_diff(
        self,
        snapshot1: SystemSnapshot,
        snapshot2: SystemSnapshot
    ) -> SnapshotDiff:
        """
        Calculate difference between snapshots

        Args:
            snapshot1: First snapshot (older)
            snapshot2: Second snapshot (newer)

        Returns:
            SnapshotDiff
        """
        components1 = set(snapshot1.components.keys())
        components2 = set(snapshot2.components.keys())

        added = list(components2 - components1)
        removed = list(components1 - components2)

        # Find modified components
        modified = {}
        common_components = components1 & components2

        for comp_name in common_components:
            state1 = snapshot1.components[comp_name]
            state2 = snapshot2.components[comp_name]

            # Compare stats
            stats_diff = {}
            for key in set(state1.stats.keys()) | set(state2.stats.keys()):
                val1 = state1.stats.get(key)
                val2 = state2.stats.get(key)
                if val1 != val2:
                    stats_diff[key] = {"old": val1, "new": val2}

            # Compare config
            config_diff = {}
            for key in set(state1.config.keys()) | set(state2.config.keys()):
                val1 = state1.config.get(key)
                val2 = state2.config.get(key)
                if val1 != val2:
                    config_diff[key] = {"old": val1, "new": val2}

            if stats_diff or config_diff:
                modified[comp_name] = {
                    "stats": stats_diff,
                    "config": config_diff
                }

        diff = SnapshotDiff(
            from_version=snapshot1.version,
            to_version=snapshot2.version,
            added_components=added,
            removed_components=removed,
            modified_components=modified
        )

        self.stats["total_diffs"] += 1

        return diff

    def restore_snapshot(
        self,
        snapshot: SystemSnapshot,
        components: Dict[str, Any]
    ) -> bool:
        """
        Restore system state from snapshot

        Args:
            snapshot: Snapshot to restore
            components: Dictionary of component instances to restore

        Returns:
            Success status
        """
        try:
            for comp_name, state in snapshot.components.items():
                if comp_name not in components:
                    continue

                component = components[comp_name]

                # Restore config
                for key, value in state.config.items():
                    if hasattr(component, key):
                        try:
                            setattr(component, key, value)
                        except AttributeError:
                            pass  # Skip read-only attributes

                # Restore stats if component has stats attribute
                if hasattr(component, 'stats') and isinstance(component.stats, dict):
                    component.stats.update(state.stats)

            self.stats["total_restorations"] += 1
            return True

        except Exception as e:
            return False

    def rollback(
        self,
        components: Dict[str, Any],
        steps: int = 1
    ) -> Optional[SystemSnapshot]:
        """
        Rollback to previous snapshot

        Args:
            components: Dictionary of component instances
            steps: Number of steps to rollback

        Returns:
            Snapshot that was restored (or None if failed)
        """
        if len(self.snapshots) < steps + 1:
            return None

        # Get snapshot to rollback to
        target_idx = len(self.snapshots) - steps - 1
        target_snapshot = self.snapshots[target_idx]

        # Restore
        success = self.restore_snapshot(target_snapshot, components)

        if success:
            self.stats["total_rollbacks"] += 1
            return target_snapshot

        return None

    def get_snapshot_by_version(self, version: str) -> Optional[SystemSnapshot]:
        """
        Get snapshot by version

        Args:
            version: Version to find

        Returns:
            Snapshot or None
        """
        for snapshot in self.snapshots:
            if snapshot.version == version:
                return snapshot
        return None

    def get_latest_snapshot(self) -> Optional[SystemSnapshot]:
        """Get latest snapshot"""
        if self.snapshots:
            return self.snapshots[-1]
        return None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all snapshots with summary info

        Returns:
            List of snapshot summaries
        """
        summaries = []
        for snapshot in self.snapshots:
            summaries.append({
                "version": snapshot.version,
                "timestamp": snapshot.timestamp.isoformat(),
                "component_count": len(snapshot.components),
                "components": list(snapshot.components.keys())
            })
        return summaries

    def clear_history(self):
        """Clear snapshot history"""
        self.snapshots.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            "current_snapshots": len(self.snapshots),
            "max_snapshots": self.max_snapshots
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_snapshots": 0,
            "total_restorations": 0,
            "total_diffs": 0,
            "total_rollbacks": 0
        }

    def export_snapshot(
        self,
        snapshot: SystemSnapshot,
        format: str = "json",
        compress: bool = False
    ) -> bytes:
        """
        Export snapshot in specified format

        Args:
            snapshot: Snapshot to export
            format: Format ("json" or "pickle")
            compress: Whether to compress

        Returns:
            Exported bytes
        """
        if format == "json":
            data = self.serialize_json(snapshot).encode('utf-8')
        elif format == "pickle":
            data = self.serialize_pickle(snapshot)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if compress:
            data = gzip.compress(data, compresslevel=9)

        return data

    def import_snapshot(
        self,
        data: bytes,
        format: str = "json",
        compressed: bool = False
    ) -> SystemSnapshot:
        """
        Import snapshot from bytes

        Args:
            data: Snapshot data
            format: Format ("json" or "pickle")
            compressed: Whether data is compressed

        Returns:
            SystemSnapshot
        """
        if compressed:
            data = gzip.decompress(data)

        if format == "json":
            snapshot = self.deserialize_json(data.decode('utf-8'))
        elif format == "pickle":
            snapshot = self.deserialize_pickle(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return snapshot
