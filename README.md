# iree-inference

IREE inference runtime bindings and target-specific model compilation.
`crates/` contains `iree-sys` (hand-written raw FFI) and `iree-runtime` (safe
wrappers). `compiler/` contains the offline ESP32-P4 tooling, and `backends/`
contains the esp-dl adapter, host reference kernels, and ESP-IDF firmware.

See the [architecture](docs/architecture.md) and the [ESP32-P4 backend
guide](backends/esp32p4/README.md) for the PP-OCRv6 build and regression commands.

The runtime, headers and Metal source compiler use the
[IREE fork snapshot after 3.14.0](https://github.com/EdwinZhanCN/iree/commit/4ff3621d17896ee7b23d246a355ad915b9219a9e).
Python tooling uses the upstream IREE 3.14.0 packages. Rebuild VMFB modules
when updating IREE because the bytecode ABI can change across versions.

See [Consume the runtime crate](docs/runtime-packages.md) for Cargo Git
dependencies, driver features and native release packaging.

## Build and test

Install Rust (the version in `rust-toolchain.toml`), CMake, Ninja, `just`,
`cargo-nextest` and `uv`. Then run:

```sh
just iree       # fetch and build the pinned IREE runtime
just testdata   # compile the test VMFB with the 3.14.0 compiler
just check
```

`just iree` fetches the pinned commit into `third_party/iree-3.14`, initializes
the required submodules, and builds into `third_party/iree-build-3.14`.
macOS also enables Metal, and all platforms enable CPU drivers. Run
`just test-gpu` on macOS for Metal tests (requires the Metal Toolchain).

`IREE_SOURCE_DIR` enables the C ABI layout checks against a local source tree.
`IREE_RUNTIME_LIB_DIR` selects a local native runtime. It accepts both
co-located libraries and the existing `just iree` build tree.

The crates are Apache-2.0 licensed. They originate from the `iree-embed`
project and earlier IREE-Rust-Exp work; see the source project for provenance.
