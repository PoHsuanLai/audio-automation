//! Minimal example showing the essential features

use audio_automation::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum Param {
    Volume,
    Cutoff,
}

fn main() {
    println!("Audio Automation - Essential Features\n");

    // 1. Create envelope with different curves
    basic_envelope();

    // 2. Preset builders
    presets();

    // 3. Mathematical operations
    math_ops();

    // 4. Iterate, index, and transform
    iterate_and_transform();

    // 5. Multi-parameter clips
    multi_params();
}

fn basic_envelope() {
    println!("1. Basic Envelope\n");

    let env = AutomationEnvelope::new(Param::Volume)
        .with_point(AutomationPoint::new(0.0, 0.0))
        .with_point(AutomationPoint::with_curve(4.0, 1.0, CurveType::SCurve));

    println!("   t=0: {:.2}", env.get_value_at(0.0).unwrap());
    println!("   t=2: {:.2}", env.get_value_at(2.0).unwrap());
    println!("   t=4: {:.2}\n", env.get_value_at(4.0).unwrap());
}

fn presets() {
    println!("2. Preset Builders\n");

    let fade = AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::Exponential);
    let lfo = AutomationEnvelope::lfo(Param::Volume, 2.0, 4.0, 0.0, 1.0);

    println!("   Fade-in at t=2: {:.2}", fade.get_value_at(2.0).unwrap());
    println!("   LFO at t=1: {:.2}\n", lfo.get_value_at(1.0).unwrap());
}

fn math_ops() {
    println!("3. Mathematical Operations\n");

    let carrier = AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::Linear);
    let lfo = AutomationEnvelope::lfo(Param::Volume, 4.0, 4.0, 0.5, 1.0);

    let am = carrier.multiply(&lfo);
    println!(
        "   AM (tremolo) at t=2: {:.2}\n",
        am.get_value_at(2.0).unwrap()
    );
}

fn iterate_and_transform() {
    println!("4. Iterate, Index, and Transform\n");

    let mut env = AutomationEnvelope::new(Param::Cutoff)
        .with_point(AutomationPoint::new(0.0, 100.0))
        .with_point(AutomationPoint::new(2.0, 800.0))
        .with_point(AutomationPoint::new(4.0, 1000.0));

    // Index into points directly
    println!("   First point: t={} v={}", env[0].time, env[0].value);

    // Iterate over points
    print!("   All points: ");
    for point in &env {
        print!("({:.0}, {:.0}) ", point.time, point.value);
    }
    println!();

    // Transform in place
    env.normalize(0.0, 1.0).scale(0.8).offset(0.1);

    println!("   After normalize→scale→offset:");
    println!("   t=0: {:.2}", env.get_value_at(0.0).unwrap());
    println!("   t=4: {:.2}\n", env.get_value_at(4.0).unwrap());
}

fn multi_params() {
    println!("5. Multi-Parameter Clips\n");

    let clip = AutomationClip::new("Intro", 8.0)
        .with_envelope(
            "volume",
            AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::Linear),
        )
        .with_envelope(
            "cutoff",
            AutomationEnvelope::new(Param::Cutoff)
                .with_point(AutomationPoint::new(0.0, 0.0))
                .with_point(AutomationPoint::new(8.0, 1.0)),
        );

    // Index into the clip by key
    println!(
        "   volume at t=2: {:.2}",
        clip["volume"].get_value_at(2.0).unwrap()
    );

    // Iterate over all envelopes
    let values = clip.get_values_at(4.0);
    for (key, value) in &clip {
        println!(
            "   {} at t=4: {:.2} (from get_values_at: {:.2})",
            key,
            value.get_value_at(4.0).unwrap(),
            values[key]
        );
    }
    println!();
}
