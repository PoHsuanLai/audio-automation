//! Automation envelope — time-indexed parameter automation with interpolation

use super::curve::CurveType;
use serde::{Deserialize, Serialize};

/// Single automation point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationPoint {
    /// Time position in beats (or seconds, depending on context)
    pub time: f64,
    /// If None, calculated from beat position and tempo map
    pub sample_position: Option<u64>,
    pub value: f32,
    /// Curve to next point
    pub curve: CurveType,
}

impl AutomationPoint {
    pub const fn new(time: f64, value: f32) -> Self {
        Self {
            time,
            sample_position: None,
            value,
            curve: CurveType::Linear,
        }
    }

    pub const fn with_curve(time: f64, value: f32, curve: CurveType) -> Self {
        Self {
            time,
            sample_position: None,
            value,
            curve,
        }
    }

    pub const fn with_samples(
        time: f64,
        sample_position: u64,
        value: f32,
        curve: CurveType,
    ) -> Self {
        Self {
            time,
            sample_position: Some(sample_position),
            value,
            curve,
        }
    }

    pub fn set_sample_position(&mut self, sample: u64) {
        self.sample_position = Some(sample);
    }
}

/// Compares two [`AutomationPoint`]s using epsilon-based floating-point comparison
/// on `time` and `value` fields. The `curve` type is not considered.
impl PartialEq for AutomationPoint {
    fn eq(&self, other: &Self) -> bool {
        (self.time - other.time).abs() < f64::EPSILON
            && (self.value - other.value).abs() < f32::EPSILON
    }
}

/// Automation envelope for a single parameter.
///
/// Generic over `T` (the automation target). Use `with_*` builder methods for construction,
/// or mutating methods (returning `&mut Self`) for chaining on an existing envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEnvelope<T> {
    pub target: T,
    /// Sorted by time
    pub points: Vec<AutomationPoint>,
    pub enabled: bool,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
    pub step_size: Option<f32>,
}

impl<T> AutomationEnvelope<T> {
    pub fn new(target: T) -> Self {
        Self {
            target,
            points: Vec::new(),
            enabled: true,
            min_value: None,
            max_value: None,
            step_size: None,
        }
    }

    pub fn with_min(mut self, min: f32) -> Self {
        self.min_value = Some(min);
        self
    }

    pub fn with_max(mut self, max: f32) -> Self {
        self.max_value = Some(max);
        self
    }

    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    pub fn with_step(mut self, step: f32) -> Self {
        self.step_size = Some(step);
        self
    }

    /// Returns `Self` (owned) for building; `add_point` returns `&mut Self` for chaining.
    pub fn with_point(mut self, point: AutomationPoint) -> Self {
        self.add_point(point);
        self
    }

    pub fn with_points(mut self, points: impl IntoIterator<Item = AutomationPoint>) -> Self {
        for point in points {
            self.add_point(point);
        }
        self
    }

    /// Inserts maintaining sorted order; replaces existing point at same time.
    pub fn add_point(&mut self, point: AutomationPoint) -> &mut Self {
        let pos = self
            .points
            .binary_search_by(|p| p.time.total_cmp(&point.time));

        match pos {
            Ok(idx) => {
                self.points[idx] = point;
            }
            Err(idx) => {
                self.points.insert(idx, point);
            }
        }
        self
    }

    pub fn remove_point_at(&mut self, time: f64) -> &mut Self {
        if let Some(pos) = self
            .points
            .iter()
            .position(|p| (p.time - time).abs() < 0.001)
        {
            self.points.remove(pos);
        }
        self
    }

    pub fn remove_point(&mut self, index: usize) -> &mut Self {
        if index < self.points.len() {
            self.points.remove(index);
        }
        self
    }

    #[must_use]
    pub fn get_point(&self, index: usize) -> Option<&AutomationPoint> {
        self.points.get(index)
    }

    pub fn get_point_mut(&mut self, index: usize) -> Option<&mut AutomationPoint> {
        self.points.get_mut(index)
    }

    pub fn clear(&mut self) -> &mut Self {
        self.points.clear();
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    #[must_use]
    #[inline]
    pub fn get_value_at(&self, time: f64) -> Option<f32> {
        if !self.enabled || self.points.is_empty() {
            return None;
        }

        if self.points.len() == 1 {
            return Some(self.apply_constraints(self.points[0].value));
        }

        if time <= self.points[0].time {
            return Some(self.apply_constraints(self.points[0].value));
        }

        let last_idx = self.points.len() - 1;
        if time >= self.points[last_idx].time {
            return Some(self.apply_constraints(self.points[last_idx].value));
        }

        let (prev_idx, next_idx) = self.find_surrounding_indices(time)?;
        let prev = &self.points[prev_idx];
        let next = &self.points[next_idx];

        let time_span = next.time - prev.time;
        let t = if time_span > 0.0 {
            ((time - prev.time) / time_span) as f32
        } else {
            0.0
        };

        let value = prev.curve.interpolate(prev.value, next.value, t);
        Some(self.apply_constraints(value))
    }

    #[must_use]
    /// Preferred for real-time audio processing — sample-accurate interpolation.
    #[inline]
    pub fn get_value_at_sample(&self, sample: u64) -> Option<f32> {
        if !self.enabled || self.points.is_empty() {
            return None;
        }

        if self.points.len() == 1 {
            return Some(self.apply_constraints(self.points[0].value));
        }

        // Fallback: if no sample positions set, treat first point as sample 0
        let first_sample = self.points[0].sample_position.unwrap_or(0);

        if sample <= first_sample {
            return Some(self.apply_constraints(self.points[0].value));
        }

        let last_idx = self.points.len() - 1;
        // Fallback: estimate based on time (assume 48kHz)
        let last_sample = self.points[last_idx]
            .sample_position
            .unwrap_or((self.points[last_idx].time * 48000.0) as u64);

        if sample >= last_sample {
            return Some(self.apply_constraints(self.points[last_idx].value));
        }

        let (prev_idx, next_idx) = self.find_surrounding_samples(sample)?;
        let prev = &self.points[prev_idx];
        let next = &self.points[next_idx];

        let prev_sample = prev.sample_position.unwrap_or((prev.time * 48000.0) as u64);
        let next_sample = next.sample_position.unwrap_or((next.time * 48000.0) as u64);

        let sample_span = next_sample - prev_sample;
        let t = if sample_span > 0 {
            (sample - prev_sample) as f32 / sample_span as f32
        } else {
            0.0
        };

        let value = prev.curve.interpolate(prev.value, next.value, t);
        Some(self.apply_constraints(value))
    }

    fn apply_constraints(&self, mut value: f32) -> f32 {
        if let Some(min) = self.min_value {
            value = value.max(min);
        }
        if let Some(max) = self.max_value {
            value = value.min(max);
        }

        if let Some(step) = self.step_size {
            if step > 0.0 {
                value = (value / step).round() * step;
            }
        }

        value
    }

    fn find_surrounding_indices(&self, time: f64) -> Option<(usize, usize)> {
        let pos = self.points.binary_search_by(|p| p.time.total_cmp(&time));

        match pos {
            Ok(exact) => Some((exact, exact)),
            Err(insert_pos) => {
                if insert_pos == 0 || insert_pos >= self.points.len() {
                    None
                } else {
                    Some((insert_pos - 1, insert_pos))
                }
            }
        }
    }

    fn find_surrounding_samples(&self, sample: u64) -> Option<(usize, usize)> {
        let pos = self.points.binary_search_by_key(&sample, |p| {
            p.sample_position.unwrap_or((p.time * 48000.0) as u64)
        });

        match pos {
            Ok(exact) => Some((exact, exact)),
            Err(insert_pos) => {
                if insert_pos == 0 || insert_pos >= self.points.len() {
                    None
                } else {
                    Some((insert_pos - 1, insert_pos))
                }
            }
        }
    }

    /// Usually not needed — `add_point` maintains sorted order.
    pub fn sort_points(&mut self) -> &mut Self {
        self.points.sort_by(|a, b| a.time.total_cmp(&b.time));
        self
    }

    pub fn validate(&mut self) {
        for i in 1..self.points.len() {
            if self.points[i].time < self.points[i - 1].time {
                self.sort_points();
                break;
            }
        }

        let mut seen_times = std::collections::HashSet::new();
        self.points.retain(|p| seen_times.insert(p.time.to_bits()));
    }

    #[must_use]
    pub fn get_range_samples(&self, start_sample: u64, end_sample: u64) -> Option<(f32, f32)> {
        if self.points.is_empty() {
            return None;
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        let sample_step = ((end_sample - start_sample) / 100).max(1);
        let mut current = start_sample;

        while current <= end_sample {
            if let Some(value) = self.get_value_at_sample(current) {
                min = min.min(value);
                max = max.max(value);
            }
            current += sample_step;
        }

        if min.is_finite() {
            Some((min, max))
        } else {
            None
        }
    }

    #[must_use]
    pub fn get_range(&self, start_time: f64, end_time: f64) -> Option<(f32, f32)> {
        if self.points.is_empty() {
            return None;
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        let sample_count = 100;
        let step = (end_time - start_time) / sample_count as f64;

        for i in 0..=sample_count {
            let time = start_time + step * i as f64;
            if let Some(value) = self.get_value_at(time) {
                min = min.min(value);
                max = max.max(value);
            }
        }

        for point in &self.points {
            if point.time >= start_time && point.time <= end_time {
                min = min.min(point.value);
                max = max.max(point.value);
            }
        }

        if min.is_finite() {
            Some((min, max))
        } else {
            None
        }
    }

    pub fn shift_points(&mut self, offset: f64) -> &mut Self {
        for point in &mut self.points {
            point.time += offset;
        }
        self
    }

    pub fn scale_time(&mut self, factor: f64) -> &mut Self {
        if factor > 0.0 {
            for point in &mut self.points {
                point.time *= factor;
            }
        }
        self
    }

    pub fn trim(&mut self, start_time: f64, end_time: f64) -> &mut Self {
        self.points
            .retain(|p| p.time >= start_time && p.time <= end_time);
        self
    }

    pub fn reverse(&mut self) -> &mut Self {
        if self.points.is_empty() {
            return self;
        }

        let max_time = self.points.last().unwrap().time;

        for point in &mut self.points {
            point.time = max_time - point.time;
        }

        self.points.reverse();
        self
    }

    pub fn invert_values(&mut self, min: f32, max: f32) -> &mut Self {
        for point in &mut self.points {
            point.value = max - (point.value - min);
        }
        self
    }

    pub fn quantize_time(&mut self, grid: f64) -> &mut Self {
        if grid <= 0.0 {
            return self;
        }

        for point in &mut self.points {
            point.time = (point.time / grid).round() * grid;
        }

        self.validate();
        self
    }

    /// Removes points whose omission would produce error ≤ `tolerance`.
    pub fn simplify(&mut self, tolerance: f32) -> &mut Self {
        if self.points.len() <= 2 {
            return self;
        }

        let mut simplified = Vec::new();
        simplified.push(self.points[0].clone());

        for i in 1..self.points.len() - 1 {
            let prev = &self.points[i - 1];
            let curr = &self.points[i];
            let next = &self.points[i + 1];

            let time_span = next.time - prev.time;
            let t = ((curr.time - prev.time) / time_span) as f32;
            let interpolated = prev.curve.interpolate(prev.value, next.value, t);

            if (curr.value - interpolated).abs() > tolerance {
                simplified.push(curr.clone());
            }
        }

        simplified.push(self.points.last().unwrap().clone());
        self.points = simplified;
        self
    }

    #[must_use]
    pub fn to_buffer(&self, sample_rate: f64, duration: f64) -> Vec<f32> {
        let num_samples = (duration * sample_rate) as usize;
        (0..num_samples)
            .map(|i| self.get_value_at(i as f64 / sample_rate).unwrap_or(0.0))
            .collect()
    }

    #[must_use]
    pub fn iter_samples(&self, sample_rate: f64, duration: f64) -> SampleIterator<'_, T> {
        SampleIterator {
            envelope: self,
            sample_rate,
            current_sample: 0,
            total_samples: (duration * sample_rate) as usize,
        }
    }

    #[must_use]
    pub fn get_slope_at(&self, time: f64) -> Option<f32> {
        if self.points.len() < 2 {
            return Some(0.0);
        }

        let delta = 0.001;
        let v1 = self.get_value_at(time - delta)?;
        let v2 = self.get_value_at(time + delta)?;

        Some((v2 - v1) / (2.0 * delta) as f32)
    }

    #[must_use]
    pub fn find_peaks(&self) -> Vec<(f64, f32)> {
        self.points
            .windows(3)
            .filter(|w| w[1].value > w[0].value && w[1].value > w[2].value)
            .map(|w| (w[1].time, w[1].value))
            .collect()
    }

    #[must_use]
    pub fn find_valleys(&self) -> Vec<(f64, f32)> {
        self.points
            .windows(3)
            .filter(|w| w[1].value < w[0].value && w[1].value < w[2].value)
            .map(|w| (w[1].time, w[1].value))
            .collect()
    }
}

impl<T: Clone> AutomationEnvelope<T> {
    /// 0.0 → 1.0 over `duration`.
    pub fn fade_in(target: T, duration: f64, curve: CurveType) -> Self {
        let mut env = Self::new(target);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::with_curve(duration, 1.0, curve));
        env
    }

    /// 1.0 → 0.0 over `duration`.
    pub fn fade_out(target: T, duration: f64, curve: CurveType) -> Self {
        let mut env = Self::new(target);
        env.add_point(AutomationPoint::new(0.0, 1.0));
        env.add_point(AutomationPoint::with_curve(duration, 0.0, curve));
        env
    }

    /// Attack → sustain → release envelope.
    pub fn pulse(target: T, fade_in: f64, sustain: f64, fade_out: f64, curve: CurveType) -> Self {
        let mut env = Self::new(target);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::with_curve(fade_in, 1.0, curve));
        env.add_point(AutomationPoint::new(fade_in + sustain, 1.0));
        env.add_point(AutomationPoint::with_curve(
            fade_in + sustain + fade_out,
            0.0,
            curve,
        ));
        env
    }

    pub fn ramp(target: T, duration: f64, start: f32, end_value: f32, curve: CurveType) -> Self {
        let mut envelope = Self::new(target);
        envelope.add_point(AutomationPoint::new(0.0, start));
        envelope.add_point(AutomationPoint::with_curve(duration, end_value, curve));
        envelope
    }

    /// Sine-wave oscillation between `min` and `max` at `frequency` Hz.
    pub fn lfo(target: T, frequency: f64, duration: f64, min: f32, max: f32) -> Self {
        let period = 1.0 / frequency;
        let num_cycles = (duration / period).ceil() as usize;

        (0..=num_cycles * 4)
            .map(|i| {
                let t = i as f64 * period / 4.0;
                let phase = (i % 4) as f32 / 4.0;
                let value =
                    min + (max - min) * ((phase * std::f32::consts::PI * 2.0).sin() * 0.5 + 0.5);
                (t, value)
            })
            .take_while(|&(t, _)| t <= duration)
            .fold(Self::new(target), |mut env, (t, value)| {
                env.add_point(AutomationPoint::new(t, value));
                env
            })
    }
}

impl<T: Clone> AutomationEnvelope<T> {
    /// Blend this envelope with another. `factor` 0.0 = this, 1.0 = other.
    #[must_use]
    pub fn blend(&self, other: &Self, factor: f32) -> Self {
        let factor = factor.clamp(0.0, 1.0);
        self.combine(other, |a, b| a * (1.0 - factor) + b * factor)
    }

    /// Concatenate `other` into this envelope, shifted by `offset`.
    pub fn merge(&mut self, other: &Self, offset: f64) -> &mut Self {
        for point in &other.points {
            self.add_point(AutomationPoint {
                time: point.time + offset,
                ..point.clone()
            });
        }
        self
    }

    /// Samples both envelopes at all time points and adds their values.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a + b)
    }

    /// Useful for amplitude modulation or applying gain curves.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a * b)
    }

    /// Useful for envelope followers or ducking effects.
    #[must_use]
    pub fn min(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a.min(b))
    }

    /// Useful for gating or ensuring minimum levels.
    #[must_use]
    pub fn max(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a.max(b))
    }

    #[must_use]
    pub fn subtract(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a - b)
    }

    fn combine<F>(&self, other: &Self, op: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        let times: std::collections::BTreeSet<u64> = self
            .points
            .iter()
            .chain(other.points.iter())
            .map(|p| p.time.to_bits())
            .collect();

        let mut result = Self::new(self.target.clone());
        for time in times.into_iter().map(f64::from_bits) {
            let v1 = self.get_value_at(time).unwrap_or(0.0);
            let v2 = other.get_value_at(time).unwrap_or(0.0);
            result.add_point(AutomationPoint::new(time, op(v1, v2)));
        }
        result
    }

    /// Scales all values to fit between `new_min` and `new_max`.
    pub fn normalize(&mut self, new_min: f32, new_max: f32) -> &mut Self {
        let (current_min, current_max) = self
            .points
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p.value), hi.max(p.value))
            });

        let range = current_max - current_min;
        if range > 0.0 {
            let new_range = new_max - new_min;
            for point in &mut self.points {
                point.value = new_min + (point.value - current_min) / range * new_range;
            }
        }

        self
    }

    pub fn scale(&mut self, factor: f32) -> &mut Self {
        for point in &mut self.points {
            point.value *= factor;
        }
        self
    }

    pub fn offset(&mut self, amount: f32) -> &mut Self {
        for point in &mut self.points {
            point.value += amount;
        }
        self
    }

    pub fn clamp_values(&mut self, min: f32, max: f32) -> &mut Self {
        for point in &mut self.points {
            point.value = point.value.clamp(min, max);
        }
        self
    }

    pub fn apply_fade_in(&mut self, duration: f64, curve: CurveType) -> &mut Self {
        if duration <= 0.0 {
            return self;
        }
        if let Some(start_time) = self.points.first().map(|p| p.time) {
            let end_time = start_time + duration;
            for point in &mut self.points {
                if point.time <= end_time {
                    let t = ((point.time - start_time) / duration).clamp(0.0, 1.0) as f32;
                    point.value *= curve.interpolate(0.0, 1.0, t);
                }
            }
        }
        self
    }

    pub fn apply_fade_out(&mut self, duration: f64, curve: CurveType) -> &mut Self {
        if duration <= 0.0 {
            return self;
        }
        if let Some(end_time) = self.points.last().map(|p| p.time) {
            let start_time = end_time - duration;
            for point in &mut self.points {
                if point.time >= start_time {
                    let t = ((point.time - start_time) / duration).clamp(0.0, 1.0) as f32;
                    point.value *= curve.interpolate(1.0, 0.0, t);
                }
            }
        }
        self
    }

    pub fn apply_fades(
        &mut self,
        fade_in_duration: f64,
        fade_out_duration: f64,
        curve: CurveType,
    ) -> &mut Self {
        self.apply_fade_in(fade_in_duration, curve)
            .apply_fade_out(fade_out_duration, curve)
    }

    pub fn apply_gate(&mut self, threshold: f32) -> &mut Self {
        for point in &mut self.points {
            point.value = if point.value < threshold {
                0.0
            } else {
                point.value
            };
        }
        self
    }

    /// Values above `threshold` are reduced by `ratio`.
    pub fn apply_compression(&mut self, threshold: f32, ratio: f32) -> &mut Self {
        for point in &mut self.points {
            if point.value > threshold {
                let excess = point.value - threshold;
                point.value = threshold + excess / ratio;
            }
        }
        self
    }
}

/// Iterator over sampled envelope values
pub struct SampleIterator<'a, T> {
    envelope: &'a AutomationEnvelope<T>,
    sample_rate: f64,
    current_sample: usize,
    total_samples: usize,
}

impl<'a, T> Iterator for SampleIterator<'a, T> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_sample >= self.total_samples {
            return None;
        }

        let time = self.current_sample as f64 / self.sample_rate;
        self.current_sample += 1;

        self.envelope.get_value_at(time)
    }
}

impl<'a, T> ExactSizeIterator for SampleIterator<'a, T> {
    fn len(&self) -> usize {
        self.total_samples - self.current_sample
    }
}

impl<T> std::ops::Index<usize> for AutomationEnvelope<T> {
    type Output = AutomationPoint;
    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl<T> std::ops::IndexMut<usize> for AutomationEnvelope<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.points[index]
    }
}

impl<'a, T> IntoIterator for &'a AutomationEnvelope<T> {
    type Item = &'a AutomationPoint;
    type IntoIter = std::slice::Iter<'a, AutomationPoint>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AutomationEnvelope<T> {
    type Item = &'a mut AutomationPoint;
    type IntoIter = std::slice::IterMut<'a, AutomationPoint>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.iter_mut()
    }
}

impl<T: Default> FromIterator<AutomationPoint> for AutomationEnvelope<T> {
    fn from_iter<I: IntoIterator<Item = AutomationPoint>>(iter: I) -> Self {
        let mut env = Self::new(T::default());
        for point in iter {
            env.add_point(point);
        }
        env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple target type for testing
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    enum TestTarget {
        Volume,
        Pan,
    }

    #[test]
    fn test_add_point_maintains_order() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(2.0, 0.5));
        env.add_point(AutomationPoint::new(1.0, 0.3));
        env.add_point(AutomationPoint::new(3.0, 0.7));

        assert_eq!(env.points.len(), 3);
        assert_eq!(env.points[0].time, 1.0);
        assert_eq!(env.points[1].time, 2.0);
        assert_eq!(env.points[2].time, 3.0);
    }

    #[test]
    fn test_get_value_at() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        // At midpoint with linear curve
        let mid_value = env.get_value_at(2.0).unwrap();
        assert!((mid_value - 0.5).abs() < 0.01);

        // Before first point
        assert_eq!(env.get_value_at(-1.0).unwrap(), 0.0);

        // After last point
        assert_eq!(env.get_value_at(5.0).unwrap(), 1.0);
    }

    #[test]
    fn test_remove_point() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(1.0, 0.1));
        env.add_point(AutomationPoint::new(2.0, 0.2));
        env.add_point(AutomationPoint::new(3.0, 0.3));

        env.remove_point_at(2.0);
        assert_eq!(env.points.len(), 2);
        assert_eq!(env.points[1].time, 3.0);
    }

    #[test]
    fn test_value_constraints() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume).with_range(0.0, 1.0);
        env.add_point(AutomationPoint::new(0.0, -0.5));
        env.add_point(AutomationPoint::new(4.0, 1.5));

        // Values should be clamped
        assert_eq!(env.get_value_at(0.0).unwrap(), 0.0);
        assert_eq!(env.get_value_at(4.0).unwrap(), 1.0);
    }

    #[test]
    fn test_step_quantization() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume).with_step(0.25);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        let value = env.get_value_at(2.0).unwrap();
        // Value should be quantized to nearest 0.25
        assert_eq!(value, 0.5);
    }

    #[test]
    fn test_shift_points() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        env.shift_points(2.0);
        assert_eq!(env.points[0].time, 2.0);
        assert_eq!(env.points[1].time, 6.0);
    }

    #[test]
    fn test_scale_time() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        env.scale_time(2.0);
        assert_eq!(env.points[0].time, 0.0);
        assert_eq!(env.points[1].time, 8.0);
    }

    #[test]
    fn test_reverse() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(2.0, 0.5));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        env.reverse();
        assert_eq!(env.points[0].time, 0.0);
        assert_eq!(env.points[0].value, 1.0);
        assert_eq!(env.points[2].time, 4.0);
        assert_eq!(env.points[2].value, 0.0);
    }

    #[test]
    fn test_invert_values() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        env.invert_values(0.0, 1.0);
        assert_eq!(env.points[0].value, 1.0);
        assert_eq!(env.points[1].value, 0.0);
    }

    #[test]
    fn test_preset_fade_in() {
        let env = AutomationEnvelope::fade_in(TestTarget::Volume, 4.0, CurveType::Linear);
        assert_eq!(env.points.len(), 2);
        assert_eq!(env.get_value_at(0.0).unwrap(), 0.0);
        assert_eq!(env.get_value_at(4.0).unwrap(), 1.0);
    }

    #[test]
    fn test_preset_pulse() {
        let env = AutomationEnvelope::pulse(TestTarget::Volume, 1.0, 2.0, 1.0, CurveType::Linear);
        assert_eq!(env.points.len(), 4);
        assert_eq!(env.get_value_at(0.0).unwrap(), 0.0);
        assert_eq!(env.get_value_at(1.0).unwrap(), 1.0);
        assert_eq!(env.get_value_at(3.0).unwrap(), 1.0);
        assert_eq!(env.get_value_at(4.0).unwrap(), 0.0);
    }

    #[test]
    fn test_to_buffer() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(1.0, 1.0));

        let buffer = env.to_buffer(10.0, 1.0);
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer[0], 0.0);
        // At time 0.9 (sample 9), should be close to 1.0
        assert!((buffer[9] - 0.9).abs() < 0.1);
    }

    #[test]
    fn test_iter_samples() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(1.0, 1.0));

        let samples: Vec<f32> = env.iter_samples(10.0, 1.0).collect();
        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0], 0.0);
        // At time 0.9 (sample 9), should be close to 1.0
        assert!((samples[9] - 0.9).abs() < 0.1);
    }

    #[test]
    fn test_find_peaks() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::new(2.0, 1.0)); // peak
        env.add_point(AutomationPoint::new(4.0, 0.5));

        let peaks = env.find_peaks();
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].0, 2.0);
        assert_eq!(peaks[0].1, 1.0);
    }

    #[test]
    fn test_blend() {
        let mut env1 = AutomationEnvelope::new(TestTarget::Volume);
        env1.add_point(AutomationPoint::new(0.0, 0.0));
        env1.add_point(AutomationPoint::new(4.0, 0.0));

        let mut env2 = AutomationEnvelope::new(TestTarget::Volume);
        env2.add_point(AutomationPoint::new(0.0, 1.0));
        env2.add_point(AutomationPoint::new(4.0, 1.0));

        let blended = env1.blend(&env2, 0.5);
        assert!((blended.get_value_at(2.0).unwrap() - 0.5).abs() < 0.01);
    }

    // ==================== Mathematical Operations Tests ====================

    #[test]
    fn test_add() {
        let mut env1 = AutomationEnvelope::new(TestTarget::Volume);
        env1.add_point(AutomationPoint::new(0.0, 0.5));
        env1.add_point(AutomationPoint::new(4.0, 0.5));

        let mut env2 = AutomationEnvelope::new(TestTarget::Volume);
        env2.add_point(AutomationPoint::new(0.0, 0.3));
        env2.add_point(AutomationPoint::new(4.0, 0.3));

        let result = env1.add(&env2);
        assert!((result.get_value_at(2.0).unwrap() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_multiply() {
        let mut env1 = AutomationEnvelope::new(TestTarget::Volume);
        env1.add_point(AutomationPoint::new(0.0, 2.0));
        env1.add_point(AutomationPoint::new(4.0, 2.0));

        let mut env2 = AutomationEnvelope::new(TestTarget::Volume);
        env2.add_point(AutomationPoint::new(0.0, 0.5));
        env2.add_point(AutomationPoint::new(4.0, 0.5));

        let result = env1.multiply(&env2);
        assert!((result.get_value_at(2.0).unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_min_max() {
        let mut env1 = AutomationEnvelope::new(TestTarget::Volume);
        env1.add_point(AutomationPoint::new(0.0, 0.3));
        env1.add_point(AutomationPoint::new(4.0, 0.3));

        let mut env2 = AutomationEnvelope::new(TestTarget::Volume);
        env2.add_point(AutomationPoint::new(0.0, 0.7));
        env2.add_point(AutomationPoint::new(4.0, 0.7));

        let min_result = env1.min(&env2);
        assert!((min_result.get_value_at(2.0).unwrap() - 0.3).abs() < 0.01);

        let max_result = env1.max(&env2);
        assert!((max_result.get_value_at(2.0).unwrap() - 0.7).abs() < 0.01);
    }

    // ==================== Normalization Tests ====================

    #[test]
    fn test_normalize() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 10.0));
        env.add_point(AutomationPoint::new(4.0, 20.0));

        env.normalize(0.0, 1.0);

        assert!((env.get_value_at(0.0).unwrap() - 0.0).abs() < 0.01);
        assert!((env.get_value_at(4.0).unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_scale_offset() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 1.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        env.scale(2.0).offset(0.5);

        assert!((env.get_value_at(2.0).unwrap() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_clamp_values() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, -1.0));
        env.add_point(AutomationPoint::new(2.0, 0.5));
        env.add_point(AutomationPoint::new(4.0, 2.0));

        env.clamp_values(0.0, 1.0);

        assert_eq!(env.points[0].value, 0.0);
        assert_eq!(env.points[1].value, 0.5);
        assert_eq!(env.points[2].value, 1.0);
    }

    // ==================== Fade Tests ====================

    #[test]
    fn test_apply_fade_in() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 1.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        env.apply_fade_in(2.0, CurveType::Linear);

        // At start should be 0 (1.0 * 0.0)
        assert!((env.get_value_at(0.0).unwrap() - 0.0).abs() < 0.01);
        // After fade should be full
        assert!((env.get_value_at(4.0).unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_fade_out() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 1.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));

        env.apply_fade_out(2.0, CurveType::Linear);

        // At start should be full
        assert!((env.get_value_at(0.0).unwrap() - 1.0).abs() < 0.01);
        // At end should be 0 (1.0 * 0.0)
        assert!((env.get_value_at(4.0).unwrap() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_fades_chaining() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 1.0));
        env.add_point(AutomationPoint::new(2.0, 1.0));
        env.add_point(AutomationPoint::new(4.0, 1.0));
        env.add_point(AutomationPoint::new(6.0, 1.0));
        env.add_point(AutomationPoint::new(8.0, 1.0));

        env.apply_fades(2.0, 2.0, CurveType::Linear);

        // Should have fade in at start and fade out at end
        assert!((env.get_value_at(0.0).unwrap() - 0.0).abs() < 0.01);
        // Middle should be close to 1.0
        assert!((env.get_value_at(4.0).unwrap() - 1.0).abs() < 0.1);
        assert!((env.get_value_at(8.0).unwrap() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_gate() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.1));
        env.add_point(AutomationPoint::new(2.0, 0.5));
        env.add_point(AutomationPoint::new(4.0, 0.2));

        env.apply_gate(0.3);

        assert_eq!(env.points[0].value, 0.0); // Below threshold
        assert_eq!(env.points[1].value, 0.5); // Above threshold
        assert_eq!(env.points[2].value, 0.0); // Below threshold
    }

    #[test]
    fn test_apply_compression() {
        let mut env = AutomationEnvelope::new(TestTarget::Volume);
        env.add_point(AutomationPoint::new(0.0, 0.5));
        env.add_point(AutomationPoint::new(2.0, 1.0));

        env.apply_compression(0.7, 2.0);

        // First point below threshold - unchanged
        assert_eq!(env.points[0].value, 0.5);
        // Second point: 0.7 + (1.0 - 0.7) / 2.0 = 0.85
        assert!((env.points[1].value - 0.85).abs() < 0.01);
    }
}
