use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn kernel_bench(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| black_box(0u64)));
}

criterion_group!(benches, kernel_bench);
criterion_main!(benches);
