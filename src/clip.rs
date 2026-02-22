//! Automation clip — container for multiple envelopes

use super::envelope::AutomationEnvelope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Groups multiple automation envelopes for unified time transformations and serialization.
///
/// # Examples
///
/// ```rust
/// use audio_automation::{AutomationClip, AutomationEnvelope, AutomationPoint, CurveType};
///
/// # #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// # enum Param { Volume, Pan }
/// let clip = AutomationClip::new("Intro", 8.0)
///     .with_envelope("volume",
///         AutomationEnvelope::new(Param::Volume)
///             .with_point(AutomationPoint::new(0.0, 0.0))
///             .with_point(AutomationPoint::new(8.0, 1.0)))
///     .with_envelope("pan",
///         AutomationEnvelope::new(Param::Pan)
///             .with_point(AutomationPoint::new(0.0, -1.0))
///             .with_point(AutomationPoint::new(8.0, 1.0)));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationClip<T> {
    pub name: String,
    pub envelopes: HashMap<String, AutomationEnvelope<T>>,
    pub duration: f64,
    pub enabled: bool,
}

impl<T> AutomationClip<T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new(name: impl Into<String>, duration: f64) -> Self {
        Self {
            name: name.into(),
            envelopes: HashMap::new(),
            duration,
            enabled: true,
        }
    }

    /// Add an envelope, returning `Self` for use during construction.
    pub fn with_envelope(
        mut self,
        key: impl Into<String>,
        envelope: AutomationEnvelope<T>,
    ) -> Self {
        self.envelopes.insert(key.into(), envelope);
        self
    }

    pub fn add_envelope(
        &mut self,
        key: impl Into<String>,
        envelope: AutomationEnvelope<T>,
    ) -> &mut Self {
        self.envelopes.insert(key.into(), envelope);
        self
    }

    #[must_use]
    pub fn get_envelope(&self, key: &str) -> Option<&AutomationEnvelope<T>> {
        self.envelopes.get(key)
    }

    pub fn get_envelope_mut(&mut self, key: &str) -> Option<&mut AutomationEnvelope<T>> {
        self.envelopes.get_mut(key)
    }

    pub fn remove_envelope(&mut self, key: &str) -> &mut Self {
        self.envelopes.remove(key);
        self
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.envelopes.keys()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.envelopes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.envelopes.is_empty()
    }

    pub fn clear(&mut self) -> &mut Self {
        self.envelopes.clear();
        self
    }

    #[must_use]
    pub fn get_values_at(&self, time: f64) -> HashMap<String, f32> {
        let mut values = HashMap::new();

        for (key, envelope) in &self.envelopes {
            if let Some(value) = envelope.get_value_at(time) {
                values.insert(key.clone(), value);
            }
        }

        values
    }

    pub fn shift_all(&mut self, offset: f64) -> &mut Self {
        for envelope in self.envelopes.values_mut() {
            envelope.shift_points(offset);
        }
        self
    }

    pub fn scale_time_all(&mut self, factor: f64) -> &mut Self {
        for envelope in self.envelopes.values_mut() {
            envelope.scale_time(factor);
        }
        self.duration *= factor;
        self
    }

    pub fn trim_all(&mut self, start_time: f64, end_time: f64) -> &mut Self {
        for envelope in self.envelopes.values_mut() {
            envelope.trim(start_time, end_time);
        }
        self.duration = end_time - start_time;
        self
    }

    pub fn reverse_all(&mut self) -> &mut Self {
        for envelope in self.envelopes.values_mut() {
            envelope.reverse();
        }
        self
    }

    pub fn quantize_all(&mut self, grid: f64) -> &mut Self {
        for envelope in self.envelopes.values_mut() {
            envelope.quantize_time(grid);
        }
        self
    }

    pub fn simplify_all(&mut self, tolerance: f32) -> &mut Self {
        for envelope in self.envelopes.values_mut() {
            envelope.simplify(tolerance);
        }
        self
    }

    #[must_use]
    pub fn duplicate(&self, new_name: impl Into<String>) -> Self {
        Self {
            name: new_name.into(),
            ..self.clone()
        }
    }

    pub fn merge_clip(&mut self, other: &Self, offset: f64) -> &mut Self {
        for (key, other_envelope) in &other.envelopes {
            if let Some(envelope) = self.envelopes.get_mut(key) {
                envelope.merge(other_envelope, offset);
            } else {
                let mut new_envelope = other_envelope.clone();
                new_envelope.shift_points(offset);
                self.envelopes.insert(key.clone(), new_envelope);
            }
        }

        let new_end = offset + other.duration;
        if new_end > self.duration {
            self.duration = new_end;
        }
        self
    }

    /// Sample all envelopes to buffers.
    #[must_use]
    pub fn to_buffers(&self, sample_rate: f64) -> HashMap<String, Vec<f32>> {
        let mut buffers = HashMap::new();

        for (key, envelope) in &self.envelopes {
            buffers.insert(key.clone(), envelope.to_buffer(sample_rate, self.duration));
        }

        buffers
    }
}

impl<T> Default for AutomationClip<T>
where
    T: Clone + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new("Untitled", 4.0)
    }
}

impl<T> std::ops::Index<&str> for AutomationClip<T>
where
    T: Clone + std::fmt::Debug,
{
    type Output = AutomationEnvelope<T>;
    fn index(&self, key: &str) -> &Self::Output {
        &self.envelopes[key]
    }
}

impl<T> std::ops::IndexMut<&str> for AutomationClip<T>
where
    T: Clone + std::fmt::Debug,
{
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.envelopes.get_mut(key).expect("key not found in clip")
    }
}

impl<'a, T> IntoIterator for &'a AutomationClip<T>
where
    T: Clone + std::fmt::Debug,
{
    type Item = (&'a String, &'a AutomationEnvelope<T>);
    type IntoIter = std::collections::hash_map::Iter<'a, String, AutomationEnvelope<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.envelopes.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AutomationClip<T>
where
    T: Clone + std::fmt::Debug,
{
    type Item = (&'a String, &'a mut AutomationEnvelope<T>);
    type IntoIter = std::collections::hash_map::IterMut<'a, String, AutomationEnvelope<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.envelopes.iter_mut()
    }
}

impl<T> IntoIterator for AutomationClip<T>
where
    T: Clone + std::fmt::Debug,
{
    type Item = (String, AutomationEnvelope<T>);
    type IntoIter = std::collections::hash_map::IntoIter<String, AutomationEnvelope<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.envelopes.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutomationPoint;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    enum TestTarget {
        Volume,
        Pan,
        Cutoff,
    }

    #[test]
    fn test_clip_creation() {
        let clip: AutomationClip<TestTarget> = AutomationClip::new("Test Clip", 8.0);
        assert_eq!(clip.name, "Test Clip");
        assert_eq!(clip.duration, 8.0);
        assert!(clip.is_empty());
    }

    #[test]
    fn test_add_and_get_envelope() {
        let mut clip = AutomationClip::new("Test", 4.0);
        let mut envelope = AutomationEnvelope::new(TestTarget::Volume);
        envelope.add_point(AutomationPoint::new(0.0, 0.0));
        envelope.add_point(AutomationPoint::new(4.0, 1.0));

        clip.add_envelope("volume", envelope);
        assert_eq!(clip.len(), 1);

        let retrieved = clip.get_envelope("volume").unwrap();
        assert_eq!(retrieved.points.len(), 2);
    }

    #[test]
    fn test_get_values_at() {
        let mut clip = AutomationClip::new("Test", 4.0);

        let mut vol = AutomationEnvelope::new(TestTarget::Volume);
        vol.add_point(AutomationPoint::new(0.0, 0.0));
        vol.add_point(AutomationPoint::new(4.0, 1.0));

        let mut pan = AutomationEnvelope::new(TestTarget::Pan);
        pan.add_point(AutomationPoint::new(0.0, -1.0));
        pan.add_point(AutomationPoint::new(4.0, 1.0));

        clip.add_envelope("volume", vol);
        clip.add_envelope("pan", pan);

        let values = clip.get_values_at(2.0);
        assert_eq!(values.len(), 2);
        assert!((values["volume"] - 0.5).abs() < 0.01);
        assert!((values["pan"] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_shift_all() {
        let mut clip = AutomationClip::new("Test", 4.0);
        let mut envelope = AutomationEnvelope::new(TestTarget::Volume);
        envelope.add_point(AutomationPoint::new(0.0, 0.0));
        envelope.add_point(AutomationPoint::new(4.0, 1.0));

        clip.add_envelope("volume", envelope);
        clip.shift_all(2.0);

        let env = clip.get_envelope("volume").unwrap();
        assert_eq!(env.points[0].time, 2.0);
        assert_eq!(env.points[1].time, 6.0);
    }

    #[test]
    fn test_merge_clip() {
        let mut clip1 = AutomationClip::new("Clip 1", 4.0);
        let mut env1 = AutomationEnvelope::new(TestTarget::Volume);
        env1.add_point(AutomationPoint::new(0.0, 0.0));
        env1.add_point(AutomationPoint::new(4.0, 1.0));
        clip1.add_envelope("volume", env1);

        let mut clip2 = AutomationClip::new("Clip 2", 4.0);
        let mut env2 = AutomationEnvelope::new(TestTarget::Volume);
        env2.add_point(AutomationPoint::new(0.0, 1.0));
        env2.add_point(AutomationPoint::new(4.0, 0.0));
        clip2.add_envelope("volume", env2);

        clip1.merge_clip(&clip2, 4.0);

        let env = clip1.get_envelope("volume").unwrap();
        // Point at 4.0 gets replaced, so we have: 0.0, 4.0 (replaced), 8.0
        assert_eq!(env.points.len(), 3);
        assert_eq!(clip1.duration, 8.0);
    }
}
