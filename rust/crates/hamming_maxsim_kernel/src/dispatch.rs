use crate::InnerKernel;

/// Pick the fastest supported inner kernel at runtime.
pub fn pick() -> InnerKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vpopcntdq") {
            return crate::avx512::hamming_min_avx512;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            return crate::avx2::hamming_min_avx2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Intentionally prefer the scalar-unrolled baseline on Apple silicon until benchmarking proves
        // a hand-written NEON path is better for 8-byte operands.
        return crate::scalar::hamming_min_scalar;
    }

    crate::scalar::hamming_min_scalar
}
