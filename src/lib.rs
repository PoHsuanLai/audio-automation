//! # audio-automation
//!
//! Generic automation system for audio parameters - framework-agnostic.
//!
//! This crate provides:
//! - **Automation curves** - Linear, exponential, logarithmic, S-curve, stepped, and bezier
//! - **Automation envelopes** - Time-based parameter control with multiple points
//! - **Automation states** - DAW-style states (Off/Play/Write/Touch/Latch)
//! - **Generic target system** - Works with any parameter type via generics
//! - **Serialization support** - Save/load automation with serde
//!
//! ## Quick Start
//!
//! ```rust
//! use audio_automation::{AutomationEnvelope, AutomationPoint, CurveType};
//!
//! // Define your own target type
//! #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
//! enum MyTarget {
//!     Volume,
//!     Pan,
//!     Cutoff,
//! }
//!
//! // Create an envelope
//! let mut envelope = AutomationEnvelope::new(MyTarget::Volume);
//!
//! // Add points
//! envelope.add_point(AutomationPoint::new(0.0, 0.0));  // Start at 0
//! envelope.add_point(AutomationPoint::with_curve(
//!     4.0,
//!     1.0,
//!     CurveType::Exponential  // Accelerating fade-in
//! ));
//!
//! // Get interpolated value at any time
//! let value = envelope.get_value_at(2.0).unwrap();  // ~0.25 (exponential curve)
//! ```
//!
//! ## Curve Types
//!
//! - **Linear** - Straight line interpolation
//! - **Exponential** - Accelerating (ease-in)
//! - **Logarithmic** - Decelerating (ease-out)
//! - **`SCurve`** - S-shaped (ease in-out)
//! - **Stepped** - No interpolation (staircase)
//! - **Bezier** - Custom curve with control points
//!
//! ## Example: Volume Fade
//!
//! ```rust
//! use audio_automation::*;
//!
//! # #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
//! # enum Target { Volume }
//!
//! let mut fade = AutomationEnvelope::new(Target::Volume);
//!
//! // Fade from 0 to 1 over 4 beats with S-curve
//! fade.add_point(AutomationPoint::new(0.0, 0.0));
//! fade.add_point(AutomationPoint::with_curve(4.0, 1.0, CurveType::SCurve));
//!
//! // Sample at different points
//! assert!(fade.get_value_at(0.0).unwrap() < 0.1);    // Near 0 at start
//! assert!(fade.get_value_at(2.0).unwrap() > 0.4);    // ~0.5 at midpoint
//! assert!(fade.get_value_at(4.0).unwrap() > 0.9);    // Near 1 at end
//! ```

pub mod clip;
pub mod curve;
pub mod envelope;
pub mod state;

pub use clip::AutomationClip;
pub use curve::CurveType;
pub use envelope::{AutomationEnvelope, AutomationPoint, SampleIterator};
pub use state::AutomationState;

/// Prelude for common imports
pub mod prelude {
    pub use crate::clip::AutomationClip;
    pub use crate::curve::CurveType;
    pub use crate::envelope::{AutomationEnvelope, AutomationPoint, SampleIterator};
    pub use crate::state::AutomationState;
}
