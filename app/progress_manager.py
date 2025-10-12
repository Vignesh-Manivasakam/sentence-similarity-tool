# app/progress_manager.py
import streamlit as st
from typing import Optional, Callable

class UnifiedProgressManager:
    """Manages progress across multiple pipeline phases"""
    
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.phase_ranges = {
            'preprocessing': (0.0, 0.15),
            'faiss_index': (0.15, 0.50),      
            'faiss_search': (0.50, 0.70),     
            'llm_analysis': (0.70, 1.0)      
        }
        self.current_phase = None
        
    def start_phase(self, phase_name: str, message: str = None):
        """Start a new phase"""
        self.current_phase = phase_name
        start_pct = self.phase_ranges[phase_name][0]
        display_message = message or f"Starting {phase_name.replace('_', ' ').title()}..."
        self.progress_bar.progress(start_pct, text=display_message)
        
    def update_phase_progress(self, phase_progress: float, message: str = None):
        """Update progress within current phase (phase_progress: 0.0 to 1.0)"""
        if not self.current_phase:
            return
            
        start_pct, end_pct = self.phase_ranges[self.current_phase]
        current_pct = start_pct + (phase_progress * (end_pct - start_pct))
        
        display_message = message or f"{self.current_phase.replace('_', ' ').title()}... {phase_progress:.0%}"
        self.progress_bar.progress(current_pct, text=display_message)
        
    def complete_phase(self, message: str = None):
        """Complete current phase"""
        if not self.current_phase:
            return
            
        end_pct = self.phase_ranges[self.current_phase][1]
        display_message = message or f"{self.current_phase.replace('_', ' ').title()} Complete ✅"
        self.progress_bar.progress(end_pct, text=display_message)
        
    def complete_all(self):
        """Mark entire pipeline as complete"""
        self.progress_bar.progress(1.0, text="🎉 All phases complete!")