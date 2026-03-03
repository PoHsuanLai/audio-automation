# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-03-04

### Changed

- Switch to `#![no_std]` with `alloc` — works on embedded and WASM targets
- Replace `std::collections::HashMap` with `hashbrown` for `no_std` compatibility
- Replace `std` math functions with `libm`
- Use `serde` with `default-features = false`

## [0.2.0] - 2026-02-22

### Added

- Chainable builder API — `with_point()`, `with_envelope()`, and mutation methods return `&mut Self`
- New `daw_workflow` example demonstrating full clip-based usage

### Changed

- Clean up API surface: remove redundant comments, make internals more idiomatic
- `keys()` now returns an iterator instead of `Vec<String>`
- `remove_envelope()` returns `&mut Self` for chaining

## [0.1.0] - 2026-01-27

### Added

- Core automation envelope with time-based parameter control
- Curve types: Linear, Exponential, Logarithmic, S-curve, Stepped, Bezier
- Automation clips for grouping multiple envelopes
- DAW-style automation states (Off/Play/Write/Touch/Latch)
- Generic target system — works with any user-defined parameter type
- Envelope operations: shift, scale, trim, reverse, quantize, simplify
- `SampleIterator` for stepping through envelope values
- Serde serialization support
- Prelude module for convenient imports
