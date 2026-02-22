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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum Param { Volume, Cutoff }

let env = AutomationEnvelope::new(Param::Volume)
    .with_point(AutomationPoint::new(0.0, 0.0))
    .with_point(AutomationPoint::with_curve(4.0, 1.0, CurveType::Exponential))
    .with_range(0.0, 1.0);

let value = env.get_value_at(2.0).unwrap();
```

## Presets

```rust
// Fade in over 4 beats
let fade = AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::SCurve);

// Pulse: 1 beat attack, 2 beat sustain, 1 beat release
let pulse = AutomationEnvelope::pulse(Param::Volume, 1.0, 2.0, 1.0, CurveType::Linear);

// 2 Hz LFO tremolo
let tremolo = AutomationEnvelope::lfo(Param::Volume, 2.0, 4.0, 0.5, 1.0);
```

## Multiple Parameters

```rust
let clip = AutomationClip::new("Intro", 8.0)
    .with_envelope("volume",
        AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::Linear))
    .with_envelope("cutoff",
        AutomationEnvelope::new(Param::Cutoff)
            .with_point(AutomationPoint::new(0.0, 0.0))
            .with_point(AutomationPoint::new(8.0, 1.0)));
```

## Curve Types

`Linear`, `Exponential`, `Logarithmic`, `SCurve`, `Stepped`, `Bezier`, `Elastic`, `Bounce`, `Back`, `Circular`, and polynomial easing variants (`QuadIn/Out/InOut`, `CubicIn/Out/InOut`, `QuartIn/Out/InOut`, `QuintIn/Out/InOut`).

## License

MIT or Apache-2.0
