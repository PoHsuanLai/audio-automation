//! Visualization example - ASCII art plots of automation curves

use audio_automation::prelude::*;

const WIDTH: usize = 60;
const HEIGHT: usize = 20;

fn main() {
    println!("\n🎨 Automation Curve Types\n");

    // Show the most important curve types
    let curves = vec![
        ("Linear", CurveType::Linear),
        ("Exponential", CurveType::Exponential),
        ("Logarithmic", CurveType::Logarithmic),
        ("S-Curve", CurveType::SCurve),
        ("Elastic", CurveType::Elastic),
        ("Bounce", CurveType::Bounce),
    ];

    for (name, curve) in curves {
        plot_curve(name, curve);
    }

    println!("\n💡 Run: cargo run --example minimal");
}

fn plot_curve(name: &str, curve_type: CurveType) {
    println!("{}", name);

    let mut grid = vec![vec![' '; WIDTH]; HEIGHT];

    // Draw axes
    for y in 0..HEIGHT {
        grid[y][0] = '│';
    }
    for x in 0..WIDTH {
        grid[HEIGHT - 1][x] = '─';
    }
    grid[HEIGHT - 1][0] = '└';

    // Sample and plot the curve
    let samples = WIDTH - 3;
    let mut prev_y = HEIGHT - 1;

    for i in 0..samples {
        let t = i as f32 / samples as f32;
        let value = curve_type.interpolate(0.0, 1.0, t);

        let x = i + 2;
        let y = HEIGHT - 2 - ((value * (HEIGHT - 3) as f32) as usize).min(HEIGHT - 3);

        if x < WIDTH && y < HEIGHT - 1 {
            grid[y][x] = '●';

            // Connect with previous point
            if i > 0 {
                let min_y = y.min(prev_y);
                let max_y = y.max(prev_y);
                for draw_y in min_y..=max_y {
                    if grid[draw_y][x] == ' ' {
                        grid[draw_y][x] = '│';
                    }
                }
            }
            prev_y = y;
        }
    }

    // Print grid
    for row in grid {
        print!("  ");
        for cell in row {
            print!("{}", cell);
        }
        println!();
    }

    // Print sample values
    print!("  ");
    for i in [0, 25, 50, 75, 100] {
        let t = i as f32 / 100.0;
        let value = curve_type.interpolate(0.0, 1.0, t);
        print!("{:.0}%={:.2}  ", i, value);
    }
    println!("\n");
}
