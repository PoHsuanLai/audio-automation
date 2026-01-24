//! Automation envelope data structures
//!
//! Defines automation points and envelopes with generic target types

use super::curve::CurveType;
use serde::{Deserialize, Serialize};

/// Single automation point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationPoint {
    /// Time position in beats (or seconds, depending on your use case)
    pub time: f64,
    /// Sample position for sample-accurate automation (NEW)
    /// If None, will be calculated from beat position and tempo map
    pub sample_position: Option<u64>,
    /// Parameter value (range depends on target)
    pub value: f32,
    /// Curve to next point
    pub curve: CurveType,
}

impl AutomationPoint {
    /// Create a new automation point with linear curve
    pub fn new(time: f64, value: f32) -> Self {
        Self {
            time,
            sample_position: None,
            value,
            curve: CurveType::Linear,
        }
    }

    /// Create a new automation point with specific curve
    pub fn with_curve(time: f64, value: f32, curve: CurveType) -> Self {
        Self {
            time,
            sample_position: None,
            value,
            curve,
        }
    }

    /// Create a new automation point with sample-accurate timing
    pub fn with_samples(time: f64, sample_position: u64, value: f32, curve: CurveType) -> Self {
        Self {
            time,
            sample_position: Some(sample_position),
            value,
            curve,
        }
    }

    /// Set sample position for sample-accurate automation
    pub fn set_sample_position(&mut self, sample: u64) {
        self.sample_position = Some(sample);
    }
}

impl PartialEq for AutomationPoint {
    fn eq(&self, other: &Self) -> bool {
        (self.time - other.time).abs() < f64::EPSILON
            && (self.value - other.value).abs() < f32::EPSILON
    }
}

/// Automation envelope for a single parameter
///
/// Generic over `T` which represents the automation target (e.g., which parameter to automate).
/// `T` should implement `Clone + Serialize + Deserialize` for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEnvelope<T> {
    /// Target parameter
    pub target: T,
    /// Automation points (sorted by time)
    pub points: Vec<AutomationPoint>,
    /// Is this envelope enabled?
    pub enabled: bool,
    /// Minimum allowed value (optional constraint)
    pub min_value: Option<f32>,
    /// Maximum allowed value (optional constraint)
    pub max_value: Option<f32>,
    /// Step size for quantized values (optional)
    pub step_size: Option<f32>,
}

impl<T> AutomationEnvelope<T> {
    /// Create a new empty envelope
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

    /// Set minimum value constraint
    pub fn with_min(mut self, min: f32) -> Self {
        self.min_value = Some(min);
        self
    }

    /// Set maximum value constraint
    pub fn with_max(mut self, max: f32) -> Self {
        self.max_value = Some(max);
        self
    }

    /// Set value range constraint
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Set step size for quantized values
    pub fn with_step(mut self, step: f32) -> Self {
        self.step_size = Some(step);
        self
    }

    /// Add a point to the envelope (maintains sorted order)
    pub fn add_point(&mut self, point: AutomationPoint) {
        // Find insertion position using binary search
        let pos = self.points.binary_search_by(|p| {
            p.time
                .partial_cmp(&point.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match pos {
            Ok(idx) => {
                // Replace existing point at same time
                self.points[idx] = point;
            }
            Err(idx) => {
                // Insert at correct position
                self.points.insert(idx, point);
            }
        }
    }

    /// Remove a point at specific time
    pub fn remove_point_at(&mut self, time: f64) -> Option<AutomationPoint> {
        let pos = self
            .points
            .iter()
            .position(|p| (p.time - time).abs() < 0.001)?;
        Some(self.points.remove(pos))
    }

    /// Remove point by index
    pub fn remove_point(&mut self, index: usize) -> Option<AutomationPoint> {
        if index < self.points.len() {
            Some(self.points.remove(index))
        } else {
            None
        }
    }

    /// Get point at specific index
    pub fn get_point(&self, index: usize) -> Option<&AutomationPoint> {
        self.points.get(index)
    }

    /// Get mutable point at specific index
    pub fn get_point_mut(&mut self, index: usize) -> Option<&mut AutomationPoint> {
        self.points.get_mut(index)
    }

    /// Clear all points
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Get number of points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if envelope is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get interpolated value at specific time
    #[inline]
    pub fn get_value_at(&self, time: f64) -> Option<f32> {
        if !self.enabled || self.points.is_empty() {
            return None;
        }

        // Single point - return its value
        if self.points.len() == 1 {
            return Some(self.apply_constraints(self.points[0].value));
        }

        // Before first point - return first value
        if time <= self.points[0].time {
            return Some(self.apply_constraints(self.points[0].value));
        }

        // After last point - return last value
        let last_idx = self.points.len() - 1;
        if time >= self.points[last_idx].time {
            return Some(self.apply_constraints(self.points[last_idx].value));
        }

        // Find surrounding points
        let (prev_idx, next_idx) = self.find_surrounding_indices(time)?;
        let prev = &self.points[prev_idx];
        let next = &self.points[next_idx];

        // Calculate interpolation factor (0.0 to 1.0)
        let time_span = next.time - prev.time;
        let t = if time_span > 0.0 {
            ((time - prev.time) / time_span) as f32
        } else {
            0.0
        };

        // Interpolate using curve type
        let value = prev.curve.interpolate(prev.value, next.value, t);
        Some(self.apply_constraints(value))
    }

    /// Get interpolated value at specific sample position (sample-accurate!)
    /// This is the preferred method for real-time audio processing
    #[inline]
    pub fn get_value_at_sample(&self, sample: u64) -> Option<f32> {
        if !self.enabled || self.points.is_empty() {
            return None;
        }

        // Single point - return its value
        if self.points.len() == 1 {
            return Some(self.apply_constraints(self.points[0].value));
        }

        // Find first point with sample_position set (or use time-based fallback)
        let first_sample = self.points[0].sample_position.unwrap_or_else(|| {
            // Fallback: if no sample positions set, treat first point as sample 0
            0
        });

        // Before first point - return first value
        if sample <= first_sample {
            return Some(self.apply_constraints(self.points[0].value));
        }

        // Find surrounding points by sample position
        let last_idx = self.points.len() - 1;
        let last_sample = self.points[last_idx].sample_position.unwrap_or_else(|| {
            // Fallback: estimate based on time
            (self.points[last_idx].time * 48000.0) as u64 // Assume 48kHz
        });

        // After last point - return last value
        if sample >= last_sample {
            return Some(self.apply_constraints(self.points[last_idx].value));
        }

        // Find surrounding points by binary search on sample positions
        let (prev_idx, next_idx) = self.find_surrounding_samples(sample)?;
        let prev = &self.points[prev_idx];
        let next = &self.points[next_idx];

        // Get sample positions (with fallback to time-based calculation)
        let prev_sample = prev.sample_position.unwrap_or_else(|| (prev.time * 48000.0) as u64);
        let next_sample = next.sample_position.unwrap_or_else(|| (next.time * 48000.0) as u64);

        // Calculate interpolation factor (0.0 to 1.0)
        let sample_span = next_sample - prev_sample;
        let t = if sample_span > 0 {
            (sample - prev_sample) as f32 / sample_span as f32
        } else {
            0.0
        };

        // Interpolate using curve type
        let value = prev.curve.interpolate(prev.value, next.value, t);
        Some(self.apply_constraints(value))
    }

    /// Apply value constraints (min, max, step)
    fn apply_constraints(&self, mut value: f32) -> f32 {
        // Apply min/max constraints
        if let Some(min) = self.min_value {
            value = value.max(min);
        }
        if let Some(max) = self.max_value {
            value = value.min(max);
        }

        // Apply step quantization
        if let Some(step) = self.step_size {
            if step > 0.0 {
                value = (value / step).round() * step;
            }
        }

        value
    }

    /// Find indices of points surrounding given time
    fn find_surrounding_indices(&self, time: f64) -> Option<(usize, usize)> {
        // Binary search for insertion position
        let pos = self.points.binary_search_by(|p| {
            p.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match pos {
            Ok(exact) => {
                // Exact match - return same index twice
                Some((exact, exact))
            }
            Err(insert_pos) => {
                if insert_pos == 0 || insert_pos >= self.points.len() {
                    None
                } else {
                    Some((insert_pos - 1, insert_pos))
                }
            }
        }
    }

    /// Find indices of points surrounding given sample position (sample-accurate!)
    fn find_surrounding_samples(&self, sample: u64) -> Option<(usize, usize)> {
        // Linear search through samples (could be optimized with binary search if needed)
        for i in 0..self.points.len() - 1 {
            let curr_sample = self.points[i].sample_position.unwrap_or_else(|| {
                (self.points[i].time * 48000.0) as u64
            });
            let next_sample = self.points[i + 1].sample_position.unwrap_or_else(|| {
                (self.points[i + 1].time * 48000.0) as u64
            });

            if sample >= curr_sample && sample <= next_sample {
                return Some((i, i + 1));
            }
        }

        None
    }

    /// Sort points by time (usually not needed as we maintain sorted order)
    pub fn sort_points(&mut self) {
        self.points.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Validate envelope (check for duplicates, sort order, etc.)
    pub fn validate(&mut self) -> bool {
        // Check if sorted
        for i in 1..self.points.len() {
            if self.points[i].time < self.points[i - 1].time {
                // Not sorted - fix it
                self.sort_points();
                break;
            }
        }

        // Check for duplicate times
        let mut seen_times = std::collections::HashSet::new();
        self.points.retain(|p| seen_times.insert(p.time.to_bits()));

        true
    }

    // ==================== Time Range Operations ====================

    /// Get minimum and maximum values within a sample range (sample-accurate!)
    pub fn get_range_samples(&self, start_sample: u64, end_sample: u64) -> Option<(f32, f32)> {
        if self.points.is_empty() {
            return None;
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;

        // Sample the envelope at regular sample intervals (e.g., every 1000 samples)
        let sample_step = ((end_sample - start_sample) / 100).max(1);
        let mut current = start_sample;

        while current <= end_sample {
            if let Some(value) = self.get_value_at_sample(current) {
                min = min.min(value);
                max = max.max(value);
            }
            current += sample_step;
        }

        if min <= max {
            Some((min, max))
        } else {
            None
        }
    }

    /// Get the minimum and maximum values within a time range
    pub fn get_range(&self, start_time: f64, end_time: f64) -> Option<(f32, f32)> {
        if self.points.is_empty() {
            return None;
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;

        // Sample the envelope at regular intervals
        let sample_count = 100;
        let step = (end_time - start_time) / sample_count as f64;

        for i in 0..=sample_count {
            let time = start_time + step * i as f64;
            if let Some(value) = self.get_value_at(time) {
                min = min.min(value);
                max = max.max(value);
            }
        }

        // Also check all points within the range
        for point in &self.points {
            if point.time >= start_time && point.time <= end_time {
                min = min.min(point.value);
                max = max.max(point.value);
            }
        }

        if min == f32::MAX || max == f32::MIN {
            None
        } else {
            Some((min, max))
        }
    }

    /// Shift all points by a time offset
    pub fn shift_points(&mut self, offset: f64) {
        for point in &mut self.points {
            point.time += offset;
        }
    }

    /// Scale time by a factor (speed up/slow down)
    pub fn scale_time(&mut self, factor: f64) {
        if factor > 0.0 {
            for point in &mut self.points {
                point.time *= factor;
            }
        }
    }

    /// Remove points outside the given time range
    pub fn trim(&mut self, start_time: f64, end_time: f64) {
        self.points
            .retain(|p| p.time >= start_time && p.time <= end_time);
    }

    // ==================== Envelope Manipulation ====================

    /// Reverse the envelope in time
    pub fn reverse(&mut self) {
        if self.points.is_empty() {
            return;
        }

        let max_time = self.points.last().unwrap().time;

        for point in &mut self.points {
            point.time = max_time - point.time;
        }

        self.points.reverse();
    }

    /// Invert values vertically within a range
    pub fn invert_values(&mut self, min: f32, max: f32) {
        for point in &mut self.points {
            point.value = max - (point.value - min);
        }
    }

    /// Quantize point times to a grid
    pub fn quantize_time(&mut self, grid: f64) {
        if grid <= 0.0 {
            return;
        }

        for point in &mut self.points {
            point.time = (point.time / grid).round() * grid;
        }

        self.validate();
    }

    /// Simplify envelope by removing redundant points
    /// tolerance: maximum allowed error when removing points
    pub fn simplify(&mut self, tolerance: f32) {
        if self.points.len() <= 2 {
            return;
        }

        let mut simplified = Vec::new();
        simplified.push(self.points[0].clone());

        for i in 1..self.points.len() - 1 {
            let prev = &self.points[i - 1];
            let curr = &self.points[i];
            let next = &self.points[i + 1];

            // Calculate what the interpolated value would be without this point
            let time_span = next.time - prev.time;
            let t = ((curr.time - prev.time) / time_span) as f32;
            let interpolated = prev.curve.interpolate(prev.value, next.value, t);

            // Keep point if error is too large
            if (curr.value - interpolated).abs() > tolerance {
                simplified.push(curr.clone());
            }
        }

        // Always keep last point
        simplified.push(self.points.last().unwrap().clone());
        self.points = simplified;
    }

    // ==================== Iteration & Sampling ====================

    /// Sample the envelope at regular intervals
    pub fn to_buffer(&self, sample_rate: f64, duration: f64) -> Vec<f32> {
        let num_samples = (duration * sample_rate) as usize;
        let mut buffer = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let time = i as f64 / sample_rate;
            buffer.push(self.get_value_at(time).unwrap_or(0.0));
        }

        buffer
    }

    /// Get an iterator over sampled values
    pub fn iter_samples(&self, sample_rate: f64, duration: f64) -> SampleIterator<'_, T> {
        SampleIterator {
            envelope: self,
            sample_rate,
            current_sample: 0,
            total_samples: (duration * sample_rate) as usize,
        }
    }

    // ==================== Analysis ====================

    /// Get the slope (rate of change) at a specific time
    pub fn get_slope_at(&self, time: f64) -> Option<f32> {
        if self.points.len() < 2 {
            return Some(0.0);
        }

        let delta = 0.001;
        let v1 = self.get_value_at(time - delta)?;
        let v2 = self.get_value_at(time + delta)?;

        Some((v2 - v1) / (2.0 * delta as f32))
    }

    /// Find local maxima (peaks) in the envelope
    pub fn find_peaks(&self) -> Vec<(f64, f32)> {
        let mut peaks = Vec::new();

        for i in 1..self.points.len() - 1 {
            let prev = &self.points[i - 1];
            let curr = &self.points[i];
            let next = &self.points[i + 1];

            if curr.value > prev.value && curr.value > next.value {
                peaks.push((curr.time, curr.value));
            }
        }

        peaks
    }

    /// Find local minima (valleys) in the envelope
    pub fn find_valleys(&self) -> Vec<(f64, f32)> {
        let mut valleys = Vec::new();

        for i in 1..self.points.len() - 1 {
            let prev = &self.points[i - 1];
            let curr = &self.points[i];
            let next = &self.points[i + 1];

            if curr.value < prev.value && curr.value < next.value {
                valleys.push((curr.time, curr.value));
            }
        }

        valleys
    }
}

// ==================== Preset Builders ====================

impl<T: Clone> AutomationEnvelope<T> {
    /// Create a fade-in envelope
    pub fn fade_in(target: T, duration: f64, curve: CurveType) -> Self {
        let mut env = Self::new(target);
        env.add_point(AutomationPoint::new(0.0, 0.0));
        env.add_point(AutomationPoint::with_curve(duration, 1.0, curve));
        env
    }

    /// Create a fade-out envelope
    pub fn fade_out(target: T, duration: f64, curve: CurveType) -> Self {
        let mut env = Self::new(target);
        env.add_point(AutomationPoint::new(0.0, 1.0));
        env.add_point(AutomationPoint::with_curve(duration, 0.0, curve));
        env
    }

    /// Create a pulse envelope (fade in, sustain, fade out)
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

    /// Create a ramp from start to end value
    pub fn ramp(target: T, duration: f64, start: f32, end_value: f32, curve: CurveType) -> Self {
        let mut envelope = Self::new(target);
        envelope.add_point(AutomationPoint::new(0.0, start));
        envelope.add_point(AutomationPoint::with_curve(duration, end_value, curve));
        envelope
    }

    /// Create an LFO (Low Frequency Oscillator) envelope
    pub fn lfo(target: T, frequency: f64, duration: f64, min: f32, max: f32) -> Self {
        let mut env = Self::new(target);
        let period = 1.0 / frequency;
        let num_cycles = (duration / period).ceil() as usize;

        for i in 0..=num_cycles * 4 {
            let t = i as f64 * period / 4.0;
            if t > duration {
                break;
            }
            let phase = (i % 4) as f32 / 4.0;
            let value = min + (max - min) * (phase * std::f32::consts::PI * 2.0).sin() * 0.5 + 0.5;
            env.add_point(AutomationPoint::new(t, value));
        }

        env
    }
}

// ==================== Envelope Blending ====================

impl<T: Clone> AutomationEnvelope<T> {
    /// Blend this envelope with another at a given factor (0.0 = this, 1.0 = other)
    pub fn blend(&self, other: &Self, factor: f32) -> Self {
        let factor = factor.clamp(0.0, 1.0);
        let mut result = Self::new(self.target.clone());

        // Collect all unique time points from both envelopes
        let mut times = std::collections::BTreeSet::new();
        for point in &self.points {
            times.insert(point.time.to_bits());
        }
        for point in &other.points {
            times.insert(point.time.to_bits());
        }

        // Sample both envelopes at all time points
        for time_bits in times {
            let time = f64::from_bits(time_bits);
            let v1 = self.get_value_at(time).unwrap_or(0.0);
            let v2 = other.get_value_at(time).unwrap_or(0.0);
            let blended = v1 * (1.0 - factor) + v2 * factor;
            result.add_point(AutomationPoint::new(time, blended));
        }

        result
    }

    /// Merge another envelope into this one (concatenate in time)
    pub fn merge(&mut self, other: &Self, offset: f64) {
        for point in &other.points {
            let mut new_point = point.clone();
            new_point.time += offset;
            self.add_point(new_point);
        }
    }

    // ==================== Mathematical Operations ====================

    /// Add another envelope's values to this one
    ///
    /// Samples both envelopes and adds their values at each time point.
    /// Returns a new envelope with the combined result.
    pub fn add(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a + b)
    }

    /// Multiply this envelope's values by another
    ///
    /// Useful for amplitude modulation or applying gain curves.
    pub fn multiply(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a * b)
    }

    /// Take the minimum value between this and another envelope at each point
    ///
    /// Useful for envelope followers or ducking effects.
    pub fn min(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a.min(b))
    }

    /// Take the maximum value between this and another envelope at each point
    ///
    /// Useful for gating or ensuring minimum levels.
    pub fn max(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a.max(b))
    }

    /// Subtract another envelope's values from this one
    pub fn subtract(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a - b)
    }

    /// Combine two envelopes using a custom function
    fn combine<F>(&self, other: &Self, op: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        let mut result = Self::new(self.target.clone());

        // Collect all unique time points from both envelopes
        let mut times = std::collections::BTreeSet::new();
        for point in &self.points {
            times.insert(point.time.to_bits());
        }
        for point in &other.points {
            times.insert(point.time.to_bits());
        }

        // Sample both envelopes at all time points
        for time_bits in times {
            let time = f64::from_bits(time_bits);
            let v1 = self.get_value_at(time).unwrap_or(0.0);
            let v2 = other.get_value_at(time).unwrap_or(0.0);
            let combined = op(v1, v2);
            result.add_point(AutomationPoint::new(time, combined));
        }

        result
    }

    // ==================== Normalization & Scaling ====================

    /// Normalize values to a specific range
    ///
    /// Scales all values to fit between `new_min` and `new_max`.
    pub fn normalize(&mut self, new_min: f32, new_max: f32) -> &mut Self {
        if self.points.is_empty() {
            return self;
        }

        // Find current min/max
        let mut current_min = f32::MAX;
        let mut current_max = f32::MIN;

        for point in &self.points {
            current_min = current_min.min(point.value);
            current_max = current_max.max(point.value);
        }

        let range = current_max - current_min;
        if range > 0.0 {
            let new_range = new_max - new_min;
            for point in &mut self.points {
                point.value = new_min + (point.value - current_min) / range * new_range;
            }
        }

        self
    }

    /// Scale all values by a factor
    pub fn scale(&mut self, factor: f32) -> &mut Self {
        for point in &mut self.points {
            point.value *= factor;
        }
        self
    }

    /// Add an offset to all values
    pub fn offset(&mut self, amount: f32) -> &mut Self {
        for point in &mut self.points {
            point.value += amount;
        }
        self
    }

    /// Clamp all values to a range
    pub fn clamp_values(&mut self, min: f32, max: f32) -> &mut Self {
        for point in &mut self.points {
            point.value = point.value.clamp(min, max);
        }
        self
    }

    // ==================== Fade Utilities ====================

    /// Apply a fade-in at the start of the envelope
    ///
    /// Multiplies the envelope by a fade-in curve for the specified duration.
    pub fn apply_fade_in(&mut self, duration: f64, curve: CurveType) -> &mut Self {
        if duration <= 0.0 || self.points.is_empty() {
            return self;
        }

        let start_time = self.points.first().map(|p| p.time).unwrap_or(0.0);
        let end_time = start_time + duration;

        // Apply fade to all points in the fade range
        for point in &mut self.points {
            if point.time <= end_time {
                let t = ((point.time - start_time) / duration).clamp(0.0, 1.0) as f32;
                let fade_value = curve.interpolate(0.0, 1.0, t);
                point.value *= fade_value;
            }
        }

        self
    }

    /// Apply a fade-out at the end of the envelope
    ///
    /// Multiplies the envelope by a fade-out curve for the specified duration.
    pub fn apply_fade_out(&mut self, duration: f64, curve: CurveType) -> &mut Self {
        if duration <= 0.0 || self.points.is_empty() {
            return self;
        }

        let end_time = self.points.last().map(|p| p.time).unwrap_or(0.0);
        let start_time = end_time - duration;

        // Apply fade to all points in the fade range
        for point in &mut self.points {
            if point.time >= start_time {
                let t = ((point.time - start_time) / duration).clamp(0.0, 1.0) as f32;
                let fade_value = curve.interpolate(1.0, 0.0, t);
                point.value *= fade_value;
            }
        }

        self
    }

    /// Apply both fade-in and fade-out
    pub fn apply_fades(
        &mut self,
        fade_in_duration: f64,
        fade_out_duration: f64,
        curve: CurveType,
    ) -> &mut Self {
        self.apply_fade_in(fade_in_duration, curve)
            .apply_fade_out(fade_out_duration, curve)
    }

    /// Apply a gate - values below threshold become zero
    pub fn apply_gate(&mut self, threshold: f32) -> &mut Self {
        for point in &mut self.points {
            if point.value < threshold {
                point.value = 0.0;
            }
        }
        self
    }

    /// Apply simple compression
    ///
    /// Values above threshold are reduced by the given ratio.
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

// ==================== Sample Iterator ====================

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
