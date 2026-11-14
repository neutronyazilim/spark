use std::hint::black_box;
use std::time::Instant;

use spark::kernel::{gemv_q4_0, gemv_q8_0, q4_0::BlockQ4_0, q8_0::BlockQ8_0};

fn main() {
    let rows = 256;
    let columns = 4096;
    let blocks = rows * columns / 32;
    let q8 = vec![
        BlockQ8_0 {
            d: 0x3c00,
            qs: [1; 32],
        };
        blocks
    ];
    let q4 = vec![
        BlockQ4_0 {
            d: 0x3c00,
            qs: [0x88; 16],
        };
        blocks
    ];
    let input = vec![0.25f32; columns];
    let mut output = vec![0.0f32; rows];
    let iterations = 100;

    let start = Instant::now();
    for _ in 0..iterations {
        gemv_q8_0(
            black_box(&q8),
            black_box(&input),
            black_box(&mut output),
            columns,
        )
        .expect("valid q8 shape");
    }
    println!(
        "q8_0: {:?} total, {:?}/iteration",
        start.elapsed(),
        start.elapsed() / iterations
    );

    let start = Instant::now();
    for _ in 0..iterations {
        gemv_q4_0(
            black_box(&q4),
            black_box(&input),
            black_box(&mut output),
            columns,
        )
        .expect("valid q4 shape");
    }
    println!(
        "q4_0: {:?} total, {:?}/iteration",
        start.elapsed(),
        start.elapsed() / iterations
    );
}
