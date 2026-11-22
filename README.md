# iree-inference

IREE inference runtime bindings and target-specific model compilation.
`crates/` contains `iree-sys` (hand-written raw FFI) and `iree-runtime` (safe
wrappers). `compiler/` contains the offline ESP32-P4 tooling, and `backends/`
contains the esp-dl adapter, host reference kernels, and ESP-IDF firmware.

See the [architecture](docs/architecture.md) and the [ESP32-P4 backend
guide](backends/esp32p4/README.md) for the PP-OCRv6 build and regression commands.

The bindings target IREE 3.13.0, commit
`89ece87b89852100c0a4ac3d515e14842d090967`. Use matching runtime headers,
static libraries and `iree-base-compiler` when building VMFB modules; VMFB
compatibility is not guaranteed across IREE versions.

## Build and test

Install Rust (the version in `rust-toolchain.toml`), CMake, Ninja, `just`,
`cargo-nextest` and `uv`. Then run:

```sh
just iree       # fetch and build the pinned IREE runtime
just testdata   # compile the test VMFB with the matching compiler
just check
```

IREE is fetched on demand, not stored as a submodule. `just iree` checks the
checkout's commit before building and initializes the IREE submodules needed
for this configuration. If an existing checkout is at a different commit,
fix or remove that checkout and retry. Runtime source and build output are
stored under the ignored `third_party/` directory. macOS also enables Metal;
CPU drivers are enabled on all platforms. Run `just test-gpu` on macOS for
Metal tests (requires the Metal Toolchain).

`IREE_SOURCE_DIR` and `IREE_RUNTIME_LIB_DIR` override the header checkout and
runtime archive directory for external builds. The latter currently expects
IREE's CMake build-tree layout for the separate `flatcc_parsing` archive.
The caller is responsible for using matching versions. Without headers or the
runtime archive, layout checks and runtime integration tests are skipped;
that configuration does not validate an IREE build.

The crates are Apache-2.0 licensed. They originate from the `iree-embed`
project and earlier IREE-Rust-Exp work; see the source project for provenance.
