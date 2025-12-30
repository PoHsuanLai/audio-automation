//! Generate DAW-style PNG plots of automation curves

use audio_automation::prelude::*;
use plotters::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum Param {
    Value,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Generating automation curve plots...\n");

    std::fs::create_dir_all("plots")?;

    plot_curve_comparison()?;
    println!("  ✓ Generated plots/curves.png");

    plot_complex_example()?;
    println!("  ✓ Generated plots/complex.png");

    plot_tremolo_example()?;
    println!("  ✓ Generated plots/tremolo.png");

    println!("\n✅ All plots generated in ./plots/\n");
    Ok(())
}

fn plot_curve_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/curves.png", (1400, 800)).into_drawing_area();
    root.fill(&RGBColor(28, 28, 32))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .build_cartesian_2d(0.0..1.0, -0.15..1.15)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .disable_axes()
        .draw()?;

    let curves = vec![
        ("Linear", CurveType::Linear, RGBColor(100, 180, 255)),
        (
            "Exponential",
            CurveType::Exponential,
            RGBColor(255, 100, 120),
        ),
        ("S-Curve", CurveType::SCurve, RGBColor(100, 230, 140)),
        ("Elastic", CurveType::Elastic, RGBColor(255, 180, 80)),
        ("Bounce", CurveType::Bounce, RGBColor(200, 120, 255)),
    ];

    for (name, curve_type, color) in curves {
        let samples: Vec<(f64, f64)> = (0..=500)
            .map(|i| {
                let t = i as f32 / 500.0;
                let value = curve_type.interpolate(0.0, 1.0, t);
                (t as f64, value as f64)
            })
            .collect();

        // Draw filled area under curve
        let mut area_samples = samples.clone();
        area_samples.insert(0, (0.0, 0.0));
        area_samples.push((1.0, 0.0));

        chart.draw_series(AreaSeries::new(area_samples, 0.0, color.mix(0.15)))?;

        // Draw the curve line
        chart
            .draw_series(LineSeries::new(
                samples,
                ShapeStyle {
                    color: color.to_rgba(),
                    filled: false,
                    stroke_width: 4,
                },
            ))?
            .label(name)
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 30, y)],
                    ShapeStyle {
                        color: color.to_rgba(),
                        filled: false,
                        stroke_width: 4,
                    },
                )
            });
    }

    chart
        .configure_series_labels()
        .background_style(RGBColor(38, 38, 42).mix(0.95))
        .border_style(RGBColor(80, 80, 85))
        .label_font(("sans-serif", 20, &RGBColor(220, 220, 225)))
        .draw()?;

    root.present()?;
    Ok(())
}

fn plot_complex_example() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/complex.png", (1400, 800)).into_drawing_area();
    root.fill(&RGBColor(28, 28, 32))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .build_cartesian_2d(0.0..8.0, 0.0..1100.0)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .disable_axes()
        .draw()?;

    // Create complex filter sweep
    let mut env = AutomationEnvelope::new(Param::Value);
    env.add_point(AutomationPoint::new(0.0, 200.0));
    env.add_point(AutomationPoint::with_curve(
        2.0,
        1000.0,
        CurveType::Exponential,
    ));
    env.add_point(AutomationPoint::with_curve(4.0, 400.0, CurveType::Bounce));
    env.add_point(AutomationPoint::with_curve(6.0, 800.0, CurveType::Elastic));
    env.add_point(AutomationPoint::with_curve(8.0, 200.0, CurveType::SCurve));

    let samples: Vec<(f64, f64)> = (0..=2000)
        .map(|i| {
            let t = i as f64 / 250.0;
            let value = env.get_value_at(t).unwrap_or(0.0) as f64;
            (t, value)
        })
        .collect();

    // Draw filled area
    let mut area_samples = samples.clone();
    area_samples.insert(0, (0.0, 0.0));
    area_samples.push((8.0, 0.0));

    chart.draw_series(AreaSeries::new(
        area_samples,
        0.0,
        RGBColor(100, 200, 255).mix(0.2),
    ))?;

    // Draw the line
    chart.draw_series(LineSeries::new(
        samples,
        ShapeStyle {
            color: RGBColor(100, 200, 255).to_rgba(),
            filled: false,
            stroke_width: 4,
        },
    ))?;

    // Draw automation points with glow
    let points: Vec<(f64, f64)> = env
        .points
        .iter()
        .map(|p| (p.time, p.value as f64))
        .collect();

    // Outer glow
    chart.draw_series(PointSeries::of_element(
        points.clone(),
        12,
        RGBColor(100, 200, 255).mix(0.3),
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;

    // Main point
    chart.draw_series(PointSeries::of_element(
        points,
        7,
        RGBColor(100, 200, 255),
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;

    root.present()?;
    Ok(())
}

fn plot_tremolo_example() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/tremolo.png", (1400, 800)).into_drawing_area();
    root.fill(&RGBColor(28, 28, 32))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .build_cartesian_2d(0.0..4.0, -0.05..1.15)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .disable_axes()
        .draw()?;

    let carrier = AutomationEnvelope::fade_in(Param::Value, 4.0, CurveType::Linear);
    let lfo = AutomationEnvelope::lfo(Param::Value, 6.0, 4.0, 0.5, 1.0);
    let tremolo = carrier.multiply(&lfo);

    // Draw carrier (dim, no fill)
    let carrier_samples: Vec<(f64, f64)> = (0..=1000)
        .map(|i| {
            let t = i as f64 / 250.0;
            let value = carrier.get_value_at(t).unwrap_or(0.0) as f64;
            (t, value)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(
            carrier_samples,
            ShapeStyle {
                color: RGBColor(100, 150, 200).mix(0.4).to_rgba(),
                filled: false,
                stroke_width: 3,
            },
        ))?
        .label("Carrier")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 30, y)],
                ShapeStyle {
                    color: RGBColor(100, 150, 200).mix(0.6).to_rgba(),
                    filled: false,
                    stroke_width: 3,
                },
            )
        });

    // Draw LFO (dim, no fill)
    let lfo_samples: Vec<(f64, f64)> = (0..=1000)
        .map(|i| {
            let t = i as f64 / 250.0;
            let value = lfo.get_value_at(t).unwrap_or(0.0) as f64;
            (t, value)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(
            lfo_samples,
            ShapeStyle {
                color: RGBColor(100, 220, 120).mix(0.4).to_rgba(),
                filled: false,
                stroke_width: 3,
            },
        ))?
        .label("LFO (6 Hz)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 30, y)],
                ShapeStyle {
                    color: RGBColor(100, 220, 120).mix(0.6).to_rgba(),
                    filled: false,
                    stroke_width: 3,
                },
            )
        });

    // Draw tremolo result with fill
    let tremolo_samples: Vec<(f64, f64)> = (0..=1000)
        .map(|i| {
            let t = i as f64 / 250.0;
            let value = tremolo.get_value_at(t).unwrap_or(0.0) as f64;
            (t, value)
        })
        .collect();

    // Fill under tremolo
    let mut area_samples = tremolo_samples.clone();
    area_samples.insert(0, (0.0, 0.0));
    area_samples.push((4.0, 0.0));

    chart.draw_series(AreaSeries::new(
        area_samples,
        0.0,
        RGBColor(255, 120, 140).mix(0.2),
    ))?;

    // Draw tremolo line
    chart
        .draw_series(LineSeries::new(
            tremolo_samples,
            ShapeStyle {
                color: RGBColor(255, 120, 140).to_rgba(),
                filled: false,
                stroke_width: 4,
            },
        ))?
        .label("Tremolo (Carrier × LFO)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 30, y)],
                ShapeStyle {
                    color: RGBColor(255, 120, 140).to_rgba(),
                    filled: false,
                    stroke_width: 4,
                },
            )
        });

    chart
        .configure_series_labels()
        .background_style(RGBColor(38, 38, 42).mix(0.95))
        .border_style(RGBColor(80, 80, 85))
        .label_font(("sans-serif", 20, &RGBColor(220, 220, 225)))
        .draw()?;

    root.present()?;
    Ok(())
}
