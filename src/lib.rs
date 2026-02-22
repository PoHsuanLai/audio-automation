//! # audio-automation
//!
//! Generic automation system for audio parameters - framework-agnostic.
//!
//! This crate provides:
//! - **Automation curves** - 20+ curve types including linear, exponential, bezier, and advanced easing
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
//! #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
//! enum Param { Volume, Cutoff }
//!
//! let env = AutomationEnvelope::new(Param::Volume)
//!     .with_point(AutomationPoint::new(0.0, 0.0))
//!     .with_point(AutomationPoint::with_curve(4.0, 1.0, CurveType::Exponential))
//!     .with_range(0.0, 1.0);
//!
//! let value = env.get_value_at(2.0).unwrap();
//! ```
//!
//! ## Curve Types
//!
//! - **Linear** - Straight line interpolation
//! - **Exponential** - Accelerating (ease-in)
//! - **Logarithmic** - Decelerating (ease-out)
//! - **SCurve** - S-shaped (ease in-out)
//! - **Stepped** - No interpolation (staircase)
//! - **Bezier** - Custom curve with control points
//! - **Advanced Easing** - Elastic, Bounce, Back, Circular, and polynomial variants
//!
//! ## Example: Presets and Transformations
//!
//! ```rust
//! use audio_automation::*;
//!
//! # #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
//! # enum Param { Volume }
//!
//! let mut automation = AutomationEnvelope::new(Param::Volume)
//!     .with_point(AutomationPoint::new(0.0, 0.0))
//!     .with_point(AutomationPoint::with_curve(4.0, 1.0, CurveType::SCurve))
//!     .with_point(AutomationPoint::new(8.0, 0.5));
//!
//! automation
//!     .shift_points(2.0)
//!     .scale_time(1.5)
//!     .clamp_values(0.0, 1.0);
//!
//! assert!(automation.get_value_at(3.0).unwrap() < 0.1);
//! ```

pub mod clip;
pub mod curve;
pub mod envelope;
pub mod state;

pub use clip::AutomationClip;
pub use curve::CurveType;
pub use envelope::{AutomationEnvelope, AutomationPoint, SampleIterator};
pub use state::AutomationState;

pub mod prelude {
    pub use crate::clip::AutomationClip;
    pub use crate::curve::CurveType;
    pub use crate::envelope::{AutomationEnvelope, AutomationPoint, SampleIterator};
    pub use crate::state::AutomationState;
}
