"""
Multi-Round Router - Support for multi-turn conversations
Phase 6: Advanced Features
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid


logger = logging.getLogger(__name__)


@dataclass
class Round:
    """A single round in a multi-round conversation"""
    round_id: int
    query: str
    selected_model: str
    confidence: float
    timestamp: datetime
    success: Optional[bool] = None
    metadata: Dict[str, Any] = None


@dataclass
class Session:
    """A multi-round conversation session"""
    session_id: str
    rounds: List[Round]
    created_at: datetime
    updated_at: datetime
    context: List[str]
    total_rounds: int


class MultiRoundRouter:
    """
    Router for multi-round conversations

    Tracks conversation history and dialogue state
    Supports model switching between rounds
    Manages round-specific context
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        max_rounds_per_session: int = 50,
        context_window: int = 10
    ):
        """
        Initialize multi-round router

        Args:
            max_sessions: Maximum number of sessions to retain
            max_rounds_per_session: Maximum rounds per session
            context_window: Number of recent rounds to keep in context
        """
        self.max_sessions = max_sessions
        self.max_rounds_per_session = max_rounds_per_session
        self.context_window = context_window

        # Session storage
        self.sessions: Dict[str, Session] = {}

        logger.info(
            f"MultiRoundRouter initialized: max_sessions={max_sessions}, "
            f"max_rounds={max_rounds_per_session}"
        )

    def create_session(self) -> str:
        """
        Create a new session

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        session = Session(
            session_id=session_id,
            rounds=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            context=[],
            total_rounds=0
        )

        self.sessions[session_id] = session

        # Maintain max sessions
        if len(self.sessions) > self.max_sessions:
            # Remove oldest session
            oldest_id = min(
                self.sessions.keys(),
                key=lambda k: self.sessions[k].created_at
            )
            del self.sessions[oldest_id]
            logger.debug(f"Removed oldest session {oldest_id}")

        logger.info(f"Created session {session_id}")

        return session_id

    def add_round(
        self,
        session_id: str,
        query: str,
        selected_model: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a round to a session

        Args:
            session_id: Session ID
            query: The query
            selected_model: Selected model
            confidence: Routing confidence
            metadata: Additional metadata

        Returns:
            Round ID
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]

        # Check max rounds
        if len(session.rounds) >= self.max_rounds_per_session:
            logger.warning(
                f"Session {session_id} reached max rounds, "
                "removing oldest round"
            )
            session.rounds.pop(0)

        # Create round
        round_id = session.total_rounds
        round_obj = Round(
            round_id=round_id,
            query=query,
            selected_model=selected_model,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # Add to session
        session.rounds.append(round_obj)
        session.total_rounds += 1
        session.updated_at = datetime.now()

        # Update context
        session.context.append(query)
        if len(session.context) > self.context_window:
            session.context = session.context[-self.context_window:]

        logger.debug(
            f"Added round {round_id} to session {session_id}: {query[:50]}..."
        )

        return round_id

    def update_round(
        self,
        session_id: str,
        round_id: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update a round with outcome

        Args:
            session_id: Session ID
            round_id: Round ID
            success: Whether round was successful
            metadata: Additional metadata
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]

        # Find round
        round_obj = None
        for r in session.rounds:
            if r.round_id == round_id:
                round_obj = r
                break

        if round_obj is None:
            raise ValueError(f"Round {round_id} not found in session {session_id}")

        # Update round
        round_obj.success = success
        if metadata:
            if round_obj.metadata is None:
                round_obj.metadata = {}
            round_obj.metadata.update(metadata)

        session.updated_at = datetime.now()

        logger.debug(f"Updated round {round_id} in session {session_id}")

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        return self.sessions.get(session_id)

    def get_context(self, session_id: str) -> List[str]:
        """Get context for a session"""
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return session.context.copy()

    def get_round_history(self, session_id: str) -> List[Round]:
        """Get all rounds for a session"""
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return session.rounds.copy()

    def should_switch_model(
        self,
        session_id: str,
        current_model: str,
        min_success_rate: float = 0.7
    ) -> bool:
        """
        Determine if model should be switched

        Args:
            session_id: Session ID
            current_model: Current model
            min_success_rate: Minimum success rate threshold

        Returns:
            True if model should be switched
        """
        session = self.sessions.get(session_id)
        if session is None or len(session.rounds) < 3:
            return False  # Not enough data

        # Check recent rounds with current model
        recent_rounds = [
            r for r in session.rounds[-5:]
            if r.selected_model == current_model and r.success is not None
        ]

        if len(recent_rounds) < 3:
            return False  # Not enough data

        # Calculate success rate
        successes = sum(1 for r in recent_rounds if r.success)
        success_rate = successes / len(recent_rounds)

        # Switch if success rate is low
        should_switch = success_rate < min_success_rate

        if should_switch:
            logger.info(
                f"Recommending model switch for session {session_id}: "
                f"{current_model} success_rate={success_rate:.2%}"
            )

        return should_switch

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for multi-round routing"""
        if not self.sessions:
            return {
                "total_sessions": 0,
                "total_rounds": 0,
                "avg_rounds_per_session": 0.0
            }

        total_rounds = sum(s.total_rounds for s in self.sessions.values())
        active_sessions = len(self.sessions)

        return {
            "total_sessions": active_sessions,
            "total_rounds": total_rounds,
            "avg_rounds_per_session": total_rounds / active_sessions if active_sessions > 0 else 0.0,
            "max_sessions": self.max_sessions,
            "max_rounds_per_session": self.max_rounds_per_session
        }

    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")

    def clear_all(self):
        """Clear all sessions"""
        self.sessions.clear()
        logger.info("Cleared all multi-round sessions")
