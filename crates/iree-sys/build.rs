//! Build script for `iree-sys`.
//!
//! 1. Links the prebuilt IREE runtime static library (`libiree_runtime_unified.a`).
//!    Build it with `just iree`, or point `IREE_RUNTIME_LIB_DIR`
//!    at a directory containing it.
//! 2. Compiles `csrc/layout_check.c` against the IREE headers so the unit tests can
//!    assert that every hand-written `#[repr(C)]` struct and constant matches the C
//!    definitions of the pinned IREE version.
//!
//! Environment:
//! - `IREE_SOURCE_DIR`      IREE checkout root (default: `<workspace>/third_party/iree-3.14`)
//! - `IREE_RUNTIME_LIB_DIR` directory with `libiree_runtime_unified.a`
//!   (default: `<workspace>/third_party/iree-build-3.14/runtime/src/iree/runtime`)
//!
//! `<workspace>/third_party` is where `just iree` clones and builds.

use std::env;
use std::path::PathBuf;

const LIB_NAME: &str = "iree_runtime_unified";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let third_party = manifest_dir.join("../../third_party");

    println!("cargo:rerun-if-env-changed=IREE_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=IREE_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-changed=csrc/layout_check.c");

    let source_dir = env::var_os("IREE_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| third_party.join("iree-3.14"));
    let include_dir = source_dir.join("runtime/src");

    let lib_dir = env::var_os("IREE_RUNTIME_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| third_party.join("iree-build-3.14/runtime/src/iree/runtime"));

    // Missing files are not watched (cargo would rerun this script, and
    // rebuild everything above it, on every build). `just iree` ends with
    // `cargo clean -p iree-sys`, so a build from before it picks the runtime up.
    let api_header = include_dir.join("iree/runtime/api.h");
    let lib_file = lib_dir.join(format!("lib{LIB_NAME}.a"));

    // --- Layout check object (headers only; does not need the runtime library). ---
    if api_header.is_file() {
        println!("cargo:rerun-if-changed={}", api_header.display());
        cc::Build::new()
            .file("csrc/layout_check.c")
            .include(&include_dir)
            // IREE's own headers trip -Wsign-compare / -Wunused-*; not our code.
            .warnings(false)
            .compile("iree_sys_layout_check");
        println!("cargo:rustc-cfg=iree_sys_layout_check");
    } else {
        println!(
            "cargo:warning=IREE headers not found at {}; layout checks disabled \
             (set IREE_SOURCE_DIR)",
            include_dir.display()
        );
    }
    println!("cargo:rustc-check-cfg=cfg(iree_sys_layout_check)");

    // --- Runtime library. ---
    if lib_file.is_file() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static={LIB_NAME}");
        println!("cargo:rerun-if-changed={}", lib_file.display());
        // The unified archive leaves out flatcc's verifier (`flatcc_verify_*`,
        // used to check bytecode modules); it is a separate archive in the
        // build tree: <build>/build_tools/third_party/flatcc/libflatcc_parsing.a,
        // where lib_dir = <build>/runtime/src/iree/runtime.
        let flatcc_dir = lib_dir
            .ancestors()
            .nth(4)
            .map(|build| build.join("build_tools/third_party/flatcc"));
        match flatcc_dir.filter(|d| d.join("libflatcc_parsing.a").is_file()) {
            Some(dir) => {
                println!("cargo:rustc-link-search=native={}", dir.display());
                println!("cargo:rustc-link-lib=static=flatcc_parsing");
            }
            None => println!(
                "cargo:warning=libflatcc_parsing.a not found next to {}; linking will \
                 fail with undefined flatcc_verify_* symbols",
                lib_file.display()
            ),
        }
        println!("cargo:root={}", lib_dir.display());
        // Read by dependents' build scripts as DEP_IREE_RUNTIME_UNIFIED_LINKED.
        println!("cargo:linked=1");
    } else {
        println!("cargo:linked=0");
        // Do not fail: `cargo check` / building rlibs does not need the library.
        // Anything that links (tests, examples, binaries) will fail with
        // undefined `iree_*` symbols until the runtime is built.
        println!(
            "cargo:warning=IREE runtime library not found at {}. Run \
             just iree or set IREE_RUNTIME_LIB_DIR. Linking \
             binaries/tests that call IREE will fail.",
            lib_file.display()
        );
    }

    // System dependencies of the runtime (local-task threads, Metal driver).
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => {
            for framework in ["Foundation", "Metal", "CoreGraphics"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        "linux" | "android" => {
            for lib in ["pthread", "dl", "m"] {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }
        _ => {}
    }
}
