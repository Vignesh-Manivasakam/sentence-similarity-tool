import streamlit as st


class UnifiedProgressManager:
    """
    Maps pipeline phase names to progress bar percentage ranges.

    Phase ranges (must stay in sync with pipeline.py phase name strings):
        preprocessing    : 0%  → 15%
        vector_index     : 15% → 50%
        similarity_search: 50% → 70%  (includes hierarchical section-aware search)
        llm_analysis     : 70% → 100%

    The similarity_search phase now supports both the legacy flat cosine search
    and the new agent-based section-aware hierarchical comparison (Skills 1-5).
    Progress within this phase is subdivided internally by pipeline.py when the
    hierarchical path is active.
    """

    def __init__(self, progress_bar) -> None:
        self.progress_bar = progress_bar
        self.phase_ranges: dict[str, tuple[float, float]] = {
            "preprocessing": (0.0, 0.15),
            "vector_index": (0.15, 0.50),
            "similarity_search": (0.50, 0.70),
            "llm_analysis": (0.70, 1.0),
        }
        self.current_phase: str | None = None

    def start_phase(self, phase_name: str, message: str | None = None) -> None:
        """Advance the bar to the start of a named phase."""
        if phase_name not in self.phase_ranges:
            raise ValueError(
                f"Unknown phase '{phase_name}'. " f"Valid phases: {list(self.phase_ranges)}"
            )
        self.current_phase = phase_name
        start_pct = self.phase_ranges[phase_name][0]
        display = message or f"Starting {phase_name.replace('_', ' ').title()}..."
        self.progress_bar.progress(start_pct, text=display)

    def update_phase_progress(self, phase_progress: float, message: str | None = None) -> None:
        """
        Update within the current phase.

        Args:
            phase_progress: A value from 0.0 to 1.0 representing progress
                            within the current phase only.
            message: Optional text shown on the progress bar.
        """
        if not self.current_phase:
            return
        start_pct, end_pct = self.phase_ranges[self.current_phase]
        current_pct = start_pct + phase_progress * (end_pct - start_pct)
        display = message or (
            f"{self.current_phase.replace('_', ' ').title()}... " f"{phase_progress:.0%}"
        )
        self.progress_bar.progress(current_pct, text=display)

    def complete_phase(self, message: str | None = None) -> None:
        """Snap the bar to the end of the current phase."""
        if not self.current_phase:
            return
        end_pct = self.phase_ranges[self.current_phase][1]
        display = message or f"{self.current_phase.replace('_', ' ').title()} Complete ✅"
        self.progress_bar.progress(end_pct, text=display)

    def complete_all(self) -> None:
        """Mark the entire pipeline as finished."""
        self.progress_bar.progress(1.0, text="🎉 All phases complete!")
