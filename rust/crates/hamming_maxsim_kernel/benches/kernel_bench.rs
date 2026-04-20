use criterion::{black_box, criterion_group, criterion_main, Criterion};

use hamming_maxsim_kernel::{dispatch, maxsim_hamming};

fn kernel_bench(c: &mut Criterion) {
    let kernel = dispatch::pick();
    let query = vec![
        black_box(0x0000_0000_0000_0003u64),
        black_box(0x0000_0000_0000_0001u64),
    ];
    let doc = vec![
        black_box(0x0000_0000_0000_0003u64),
        black_box(0x0000_0000_0000_0000u64),
    ];
    c.bench_function("maxsim_hamming_small_workload", |b| {
        b.iter(|| maxsim_hamming(black_box(&query), black_box(&doc), kernel))
    });
}

criterion_group!(benches, kernel_bench);
criterion_main!(benches);
