# Consume the runtime crate

After the native profile is published, add the crate from a Git revision
using stable Cargo. This example enables CPU and Metal on Apple Silicon.

```toml
[dependencies]
iree-runtime = { git = "https://github.com/EdwinZhanCN/iree-inference.git", rev = "<iree-inference-commit>", features = ["half", "metal"] }
```

CPU drivers are included in every profile. The `metal`, `vulkan` and `cuda`
features select native driver support. `half` enables Rust FP16 and BF16 tensor
types. `iree-sys` selects native files using Cargo's target triple and
the enabled driver features.

| Profile | Release build targets |
| --- | --- |
| CPU | macOS, Linux GNU and Windows MSVC on x86-64 and ARM64 |
| Metal | macOS ARM64 |
| Vulkan | Linux GNU and Windows MSVC on x86-64 and ARM64 |
| CUDA and Vulkan + CUDA | Linux GNU and Windows MSVC on x86-64 and ARM64 |

The IREE snapshot's CUDA toolkit fetcher has no Windows ARM64 package. The
runtime build needs only CUDA headers and libdevice, which do not depend on the
platform, so the Windows ARM64 build takes them from the Windows x64 package of
the same CUDA version. GPU inference on the target driver is qualified
separately from the release job's CPU link and VMFB smoke test.

## Where the native files come from

The crate pins the IREE fork snapshot in `crates/iree-sys/runtime-revision`.
The `runtime-v1-<fork-commit>-<profile>` release serves two static libraries
and a JSON manifest for each target. The manifest identifies the fork revision,
target and driver profile and carries the SHA-256 of both libraries. The
build script validates those fields before linking and caches the files in
Cargo's build output directory.

For a local or offline build, set `IREE_RUNTIME_ASSET_DIR` to a directory with
the release files and their JSON manifest. Set `IREE_RUNTIME_LIB_DIR` to use
raw local libraries. In an `iree-inference` source checkout, the runtime from
`just iree` is used when its driver configuration matches the requested
profile. Set `CARGO_NET_OFFLINE=true` to prevent downloads during the build.

## Publish a native profile

Run the [Native runtime release workflow](../.github/workflows/runtime-release.yml)
with the desired profile, and optionally one target. The workflow compiles the native libraries, packages
them with a manifest and validates each target with a clean external Git
dependency using stable Cargo. Set the workflow's publish input to publish
the verified assets to GitHub Releases. When the profile's release exists,
the new target's files are added to it and the existing files stay.

The packaging and smoke test also run locally. For example, after
`just iree` and `just testdata` on Apple Silicon:

```sh
python3 tools/runtime-release.py package metal aarch64-apple-darwin /tmp/iree-assets --package-only
python3 tools/runtime-release.py smoke metal /tmp/iree-assets testdata/build/basic.metal.vmfb --driver metal
```
