//! Minimal example showing the essential features

use audio_automation::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum Param {
    Volume,
    Cutoff,
}

fn main() {
    println!("🎵 Audio Automation - Essential Features\n");

    // 1. Create envelope with different curves
    basic_envelope();

    // 2. Preset builders
    presets();

    // 3. Mathematical operations
    math_ops();

    // 4. Method chaining
    chaining();
}

fn basic_envelope() {
    println!("1️⃣  Basic Envelope\n");

    let mut env = AutomationEnvelope::new(Param::Volume);
    env.add_point(AutomationPoint::new(0.0, 0.0));
    env.add_point(AutomationPoint::with_curve(4.0, 1.0, CurveType::SCurve));

    println!("   t=0: {:.2}", env.get_value_at(0.0).unwrap());
    println!("   t=2: {:.2}", env.get_value_at(2.0).unwrap());
    println!("   t=4: {:.2}\n", env.get_value_at(4.0).unwrap());
}

fn presets() {
    println!("2️⃣  Preset Builders\n");

    let fade = AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::Exponential);
    let lfo = AutomationEnvelope::lfo(Param::Volume, 2.0, 4.0, 0.0, 1.0);

    println!("   Fade-in at t=2: {:.2}", fade.get_value_at(2.0).unwrap());
    println!("   LFO at t=1: {:.2}\n", lfo.get_value_at(1.0).unwrap());
}

fn math_ops() {
    println!("3️⃣  Mathematical Operations\n");

    let carrier = AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::Linear);
    let lfo = AutomationEnvelope::lfo(Param::Volume, 4.0, 4.0, 0.5, 1.0);

    let am = carrier.multiply(&lfo);
    println!(
        "   AM (tremolo) at t=2: {:.2}\n",
        am.get_value_at(2.0).unwrap()
    );
}

fn chaining() {
    println!("4️⃣  Method Chaining\n");

    let mut env = AutomationEnvelope::new(Param::Cutoff);
    env.add_point(AutomationPoint::new(0.0, 100.0));
    env.add_point(AutomationPoint::new(4.0, 1000.0));

    env.normalize(0.0, 1.0)
        .scale(0.8)
        .offset(0.1)
        .apply_fade_in(1.0, CurveType::Linear);

    println!("   After normalize→scale→offset→fade:");
    println!("   t=0: {:.2}", env.get_value_at(0.0).unwrap());
    println!("   t=4: {:.2}", env.get_value_at(4.0).unwrap());
}
