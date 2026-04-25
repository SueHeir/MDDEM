//! GPU smoke test: same setup as `granular_basic`, but `zero_all_forces`,
//! velocity-Verlet integration, and quaternion rotation run on the GPU via
//! `mddem_gpu`.
//!
//! ```bash
//! # Wgpu (Metal / Vulkan / DX12) — f32 precision, runs on any consumer GPU.
//! cargo run --example granular_basic_gpu --no-default-features --features gpu \
//!     -- examples/granular_basic_gpu/config.toml wgpu
//!
//! # cubecl CPU runtime — f64, available on any platform.
//! cargo run --example granular_basic_gpu --no-default-features --features gpu \
//!     -- examples/granular_basic_gpu/config.toml cpu
//! ```
//!
//! Compare the per-system timing breakdown against `granular_basic` to see
//! the effect of the per-step host↔device round-trips. Per-step physics
//! should match within ~1e-5 (f32 / wgpu) or ~1e-15 (f64 / cpu).

use mddem::prelude::*;

fn main() {
    let backend_arg = std::env::args().nth(2).unwrap_or_else(|| "wgpu".into());
    let backend = match backend_arg.as_str() {
        "wgpu" => BackendKind::Wgpu,
        "cpu" => BackendKind::Cpu,
        other => panic!("unknown backend '{other}'; use 'wgpu' or 'cpu'"),
    };

    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GpuDefaultPlugins::new(backend));
    app.start();
}
