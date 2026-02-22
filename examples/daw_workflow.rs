//! DAW-style workflow: arrange clips, gate with automation state, render to audio

use audio_automation::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum Param {
    Volume,
    Cutoff,
}

fn main() {
    println!("DAW Workflow\n");

    arrange_clips();
    automation_states();
    render_to_audio();
}

/// Stitch two clips together end-to-end, like arranging sections in a timeline.
fn arrange_clips() {
    println!("1. Arranging Clips\n");

    let intro = AutomationClip::new("Intro", 4.0)
        .with_envelope(
            "volume",
            AutomationEnvelope::fade_in(Param::Volume, 4.0, CurveType::Linear),
        )
        .with_envelope(
            "cutoff",
            AutomationEnvelope::new(Param::Cutoff)
                .with_point(AutomationPoint::new(0.0, 0.2))
                .with_point(AutomationPoint::new(4.0, 0.2)),
        );

    let mut verse = AutomationClip::new("Verse", 8.0)
        .with_envelope(
            "volume",
            AutomationEnvelope::new(Param::Volume)
                .with_point(AutomationPoint::new(0.0, 1.0))
                .with_point(AutomationPoint::new(8.0, 1.0)),
        )
        .with_envelope(
            "cutoff",
            AutomationEnvelope::new(Param::Cutoff)
                .with_point(AutomationPoint::new(0.0, 0.2))
                .with_point(AutomationPoint::with_curve(
                    8.0,
                    1.0,
                    CurveType::Exponential,
                )),
        );

    // Place intro at the start of verse's timeline
    verse.merge_clip(&intro, 0.0);

    println!("   Timeline duration: {:.1} beats", verse.duration);
    println!(
        "   volume at t=0 (fade start):  {:.2}",
        verse["volume"].get_value_at(0.0).unwrap()
    );
    println!(
        "   volume at t=2 (mid fade-in): {:.2}",
        verse["volume"].get_value_at(2.0).unwrap()
    );
    println!(
        "   volume at t=4 (verse body):  {:.2}",
        verse["volume"].get_value_at(4.0).unwrap()
    );
    println!(
        "   cutoff at t=8 (end of sweep): {:.2}\n",
        verse["cutoff"].get_value_at(8.0).unwrap()
    );
}

/// AutomationState controls whether a parameter lane reads or records automation.
fn automation_states() {
    println!("2. Automation States\n");

    let env = AutomationEnvelope::new(Param::Volume)
        .with_point(AutomationPoint::new(0.0, 0.0))
        .with_point(AutomationPoint::new(4.0, 1.0));

    let states = AutomationState::all();
    for &state in states {
        let reads = state.reads_automation();
        let records = state.can_record();

        // Simulate what a DAW parameter lane would do each state
        let effective_value = if reads {
            env.get_value_at(2.0).unwrap() // read from curve
        } else {
            0.75 // use manual knob position
        };

        println!(
            "   [{:3}]  reads={:5}  records={:5}  → value at t=2: {:.2}",
            state.abbreviation(),
            reads,
            records,
            effective_value
        );
    }
    println!();
}

/// Convert an envelope to a sample buffer — the final step before handing off to audio.
fn render_to_audio() {
    println!("3. Rendering to Audio\n");

    let sample_rate = 48_000.0;
    let duration = 1.0; // 1 second

    let env = AutomationEnvelope::new(Param::Volume)
        .with_point(AutomationPoint::new(0.0, 0.0))
        .with_point(AutomationPoint::with_curve(
            duration,
            1.0,
            CurveType::SCurve,
        ));

    // Option A: collect into a Vec<f32> for offline rendering
    let buffer = env.to_buffer(sample_rate, duration);
    println!("   Buffer length: {} samples", buffer.len());
    println!("   buffer[0]:      {:.4}", buffer[0]);
    println!("   buffer[24000]:  {:.4}", buffer[24_000]);
    println!("   buffer[47999]:  {:.4}", buffer[47_999]);

    // Option B: SampleIterator for streaming / real-time processing
    let rms = {
        let sum_sq: f32 = env.iter_samples(sample_rate, duration).map(|v| v * v).sum();
        (sum_sq / buffer.len() as f32).sqrt()
    };
    println!("   RMS over 1 s:   {:.4}\n", rms);
}
