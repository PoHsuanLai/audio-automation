//! Automation State Machine
//!
//! Implements DAW-style automation states that control how automation
//! is recorded and played back for each parameter lane.
//!
//! | State | Description |
//! |-------|-------------|
//! | Off   | Ignores automation, uses manual value only |
//! | Play  | Reads automation curve, no recording |
//! | Write | Records continuously, overwrites all existing automation |
//! | Touch | Records only while touching the control, returns to existing curve |
//! | Latch | Records while touching, continues at last value when released |
//!
//! These states are standard across professional DAWs (Ardour, Pro Tools, Logic, etc.)

use serde::{Deserialize, Serialize};

/// Automation state for a parameter lane (Ardour/Pro Tools style)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum AutomationState {
    #[default]
    Off,
    Play,
    Write,
    Touch,
    Latch,
}

impl AutomationState {
    #[must_use]
    #[inline]
    pub fn reads_automation(&self) -> bool {
        matches!(self, Self::Play | Self::Touch | Self::Latch)
    }

    #[must_use]
    #[inline]
    pub fn can_record(&self) -> bool {
        matches!(self, Self::Write | Self::Touch | Self::Latch)
    }

    #[must_use]
    #[inline]
    pub fn starts_on_touch(&self) -> bool {
        matches!(self, Self::Touch | Self::Latch)
    }

    #[must_use]
    #[inline]
    pub fn stops_on_release(&self) -> bool {
        matches!(self, Self::Touch)
    }

    #[must_use]
    pub fn all() -> &'static [AutomationState] {
        &[Self::Off, Self::Play, Self::Write, Self::Touch, Self::Latch]
    }

    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Play => "Play",
            Self::Write => "Write",
            Self::Touch => "Touch",
            Self::Latch => "Latch",
        }
    }

    #[must_use]
    pub fn abbreviation(&self) -> &'static str {
        match self {
            Self::Off => "OFF",
            Self::Play => "PLY",
            Self::Write => "WRT",
            Self::Touch => "TCH",
            Self::Latch => "LCH",
        }
    }
}

impl std::fmt::Display for AutomationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automation_state_properties() {
        // Off: no read, no record
        assert!(!AutomationState::Off.reads_automation());
        assert!(!AutomationState::Off.can_record());

        // Play: read only
        assert!(AutomationState::Play.reads_automation());
        assert!(!AutomationState::Play.can_record());

        // Write: record only (overwrites)
        assert!(!AutomationState::Write.reads_automation());
        assert!(AutomationState::Write.can_record());

        // Touch: read + record on touch, stops on release
        assert!(AutomationState::Touch.reads_automation());
        assert!(AutomationState::Touch.can_record());
        assert!(AutomationState::Touch.starts_on_touch());
        assert!(AutomationState::Touch.stops_on_release());

        // Latch: read + record on touch, continues after release
        assert!(AutomationState::Latch.reads_automation());
        assert!(AutomationState::Latch.can_record());
        assert!(AutomationState::Latch.starts_on_touch());
        assert!(!AutomationState::Latch.stops_on_release());
    }

    #[test]
    fn test_all_states() {
        assert_eq!(AutomationState::all().len(), 5);
    }

    #[test]
    fn test_display() {
        assert_eq!(AutomationState::Touch.display_name(), "Touch");
        assert_eq!(AutomationState::Touch.abbreviation(), "TCH");
        assert_eq!(format!("{}", AutomationState::Touch), "Touch");
    }

    #[test]
    fn test_default() {
        assert_eq!(AutomationState::default(), AutomationState::Off);
    }
}
