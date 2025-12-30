# audio-automation

[![CI](https://github.com/PoHsuanLai/audio-automation/workflows/CI/badge.svg)](https://github.com/PoHsuanLai/audio-automation/actions)
[![Crates.io](https://img.shields.io/crates/v/audio-automation.svg)](https://crates.io/crates/audio-automation)
[![docs.rs](https://docs.rs/audio-automation/badge.svg)](https://docs.rs/audio-automation)
[![License](https://img.shields.io/crates/l/audio-automation.svg)](LICENSE-MIT)

Time-based parameter automation with interpolation curves for audio applications.

![Automation Curves](plots/complex.png)

## Usage

```rust
use audio_automation::prelude::*;

let mut env = AutomationEnvelope::new("volume");
env.add_point(AutomationPoint::new(0.0, 0.0));
env.add_point(AutomationPoint::with_curve(4.0, 1.0, CurveType::Exponential));

let value = env.get_value_at(2.0).unwrap(); // interpolated value
```

## Curve types

`Linear`, `Exponential`, `Logarithmic`, `SCurve`, `Stepped`, `Bezier`, `Elastic`, `Bounce`, `Back`, `Circular`, and polynomial easing variants.

## License

MIT or Apache-2.0
