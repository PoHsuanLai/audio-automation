//! Curve interpolation for automation
//!
//! Provides different interpolation curves for smooth automation

use serde::{Deserialize, Serialize};

/// Automation curve type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum CurveType {
    /// Straight line interpolation
    #[default]
    Linear,
    /// Exponential curve (accelerating)
    Exponential,
    /// Logarithmic curve (decelerating)
    Logarithmic,
    /// S-shaped curve (ease in-out)
    SCurve,
    /// No interpolation (stepped/staircase)
    Stepped,
    /// Cubic bezier with control points
    Bezier(f32, f32),

    // Advanced easing functions
    /// Elastic bounce at end (overshoot and settle)
    Elastic,
    /// Bounce effect (like a ball bouncing)
    Bounce,
    /// Back easing (pull back before going forward)
    Back,
    /// Circular easing (quarter circle)
    Circular,
    /// Quadratic ease in
    QuadIn,
    /// Quadratic ease out
    QuadOut,
    /// Quadratic ease in-out
    QuadInOut,
    /// Cubic ease in
    CubicIn,
    /// Cubic ease out
    CubicOut,
    /// Cubic ease in-out
    CubicInOut,
    /// Quartic ease in
    QuartIn,
    /// Quartic ease out
    QuartOut,
    /// Quartic ease in-out
    QuartInOut,
    /// Quintic ease in
    QuintIn,
    /// Quintic ease out
    QuintOut,
    /// Quintic ease in-out
    QuintInOut,
}

impl CurveType {
    /// Interpolate between two values at time t (0.0 to 1.0)
    #[inline]
    pub fn interpolate(&self, start: f32, end: f32, t: f32) -> f32 {
        // Clamp t to [0, 1]
        let t = t.clamp(0.0, 1.0);

        let t_eased = match self {
            CurveType::Linear => t,
            CurveType::Exponential => t * t,
            CurveType::Logarithmic => t.sqrt(),
            CurveType::SCurve => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            CurveType::Stepped => {
                if t < 1.0 {
                    return start;
                }
                return end;
            }
            CurveType::Bezier(cp1, cp2) => cubic_bezier(t, *cp1, *cp2),

            // Advanced easing functions
            CurveType::Elastic => {
                if t == 0.0 {
                    0.0
                } else if t == 1.0 {
                    1.0
                } else {
                    let p = 0.3;
                    let s = p / 4.0;
                    -(2.0_f32.powf(10.0 * (t - 1.0)))
                        * ((t - 1.0 - s) * (2.0 * std::f32::consts::PI) / p).sin()
                }
            }
            CurveType::Bounce => {
                let t = 1.0 - t;
                let result = if t < 1.0 / 2.75 {
                    7.5625 * t * t
                } else if t < 2.0 / 2.75 {
                    let t = t - 1.5 / 2.75;
                    7.5625 * t * t + 0.75
                } else if t < 2.5 / 2.75 {
                    let t = t - 2.25 / 2.75;
                    7.5625 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / 2.75;
                    7.5625 * t * t + 0.984_375
                };
                1.0 - result
            }
            CurveType::Back => {
                let s = 1.70158;
                t * t * ((s + 1.0) * t - s)
            }
            CurveType::Circular => 1.0 - (1.0 - t * t).sqrt(),

            // Quadratic
            CurveType::QuadIn => t * t,
            CurveType::QuadOut => 1.0 - (1.0 - t) * (1.0 - t),
            CurveType::QuadInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }

            // Cubic
            CurveType::CubicIn => t * t * t,
            CurveType::CubicOut => {
                let t = 1.0 - t;
                1.0 - t * t * t
            }
            CurveType::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let t = -2.0 * t + 2.0;
                    1.0 - t * t * t / 2.0
                }
            }

            // Quartic
            CurveType::QuartIn => t * t * t * t,
            CurveType::QuartOut => {
                let t = 1.0 - t;
                1.0 - t * t * t * t
            }
            CurveType::QuartInOut => {
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    let t = -2.0 * t + 2.0;
                    1.0 - t * t * t * t / 2.0
                }
            }

            // Quintic
            CurveType::QuintIn => t * t * t * t * t,
            CurveType::QuintOut => {
                let t = 1.0 - t;
                1.0 - t * t * t * t * t
            }
            CurveType::QuintInOut => {
                if t < 0.5 {
                    16.0 * t * t * t * t * t
                } else {
                    let t = -2.0 * t + 2.0;
                    1.0 - t * t * t * t * t / 2.0
                }
            }
        };

        start + (end - start) * t_eased
    }

    /// Get a descriptive name for this curve type
    pub fn name(&self) -> &'static str {
        match self {
            CurveType::Linear => "Linear",
            CurveType::Exponential => "Exponential",
            CurveType::Logarithmic => "Logarithmic",
            CurveType::SCurve => "S-Curve",
            CurveType::Stepped => "Stepped",
            CurveType::Bezier(_, _) => "Bezier",
            CurveType::Elastic => "Elastic",
            CurveType::Bounce => "Bounce",
            CurveType::Back => "Back",
            CurveType::Circular => "Circular",
            CurveType::QuadIn => "Quad In",
            CurveType::QuadOut => "Quad Out",
            CurveType::QuadInOut => "Quad In-Out",
            CurveType::CubicIn => "Cubic In",
            CurveType::CubicOut => "Cubic Out",
            CurveType::CubicInOut => "Cubic In-Out",
            CurveType::QuartIn => "Quart In",
            CurveType::QuartOut => "Quart Out",
            CurveType::QuartInOut => "Quart In-Out",
            CurveType::QuintIn => "Quint In",
            CurveType::QuintOut => "Quint Out",
            CurveType::QuintInOut => "Quint In-Out",
        }
    }
}

/// Cubic bezier interpolation
/// Given t (0 to 1) and two control points, returns interpolated t
#[inline]
fn cubic_bezier(t: f32, cp1: f32, cp2: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    // Cubic bezier formula: B(t) = (1-t)³ + 3(1-t)²t·cp1 + 3(1-t)t²·cp2 + t³
    mt3 + 3.0 * mt2 * t * cp1 + 3.0 * mt * t2 * cp2 + t3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let curve = CurveType::Linear;
        assert_eq!(curve.interpolate(0.0, 100.0, 0.0), 0.0);
        assert_eq!(curve.interpolate(0.0, 100.0, 0.5), 50.0);
        assert_eq!(curve.interpolate(0.0, 100.0, 1.0), 100.0);
    }

    #[test]
    fn test_stepped() {
        let curve = CurveType::Stepped;
        assert_eq!(curve.interpolate(0.0, 100.0, 0.0), 0.0);
        assert_eq!(curve.interpolate(0.0, 100.0, 0.99), 0.0);
        assert_eq!(curve.interpolate(0.0, 100.0, 1.0), 100.0);
    }

    #[test]
    fn test_exponential() {
        let curve = CurveType::Exponential;
        let mid = curve.interpolate(0.0, 100.0, 0.5);
        // Exponential should be less than linear at midpoint
        assert!(mid < 50.0);
    }

    #[test]
    fn test_logarithmic() {
        let curve = CurveType::Logarithmic;
        let mid = curve.interpolate(0.0, 100.0, 0.5);
        // Logarithmic should be more than linear at midpoint
        assert!(mid > 50.0);
    }
}
