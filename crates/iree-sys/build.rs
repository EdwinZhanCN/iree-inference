//! Link the pinned IREE runtime and check FFI layouts in source builds.

#[path = "build/runtime_assets.rs"]
mod runtime_assets;

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use runtime_assets::{LIBRARIES, library_filename};

fn main() {
    if let Err(err) = build() {
        panic!("IREE runtime setup failed: {err}");
    }
}

fn build() -> Result<(), Box<dyn std::error::Error>> {
    for name in [
        "IREE_SOURCE_DIR",
        "IREE_RUNTIME_LIB_DIR",
        "IREE_RUNTIME_ASSET_DIR",
        "CARGO_NET_OFFLINE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=runtime-revision");
    println!("cargo:rerun-if-changed=csrc/layout_check.c");
    println!("cargo:rustc-check-cfg=cfg(iree_sys_layout_check)");

    let third_party =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../third_party");
    let source = env::var_os("IREE_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| third_party.join("iree-3.14"));
    check_layout(&source);

    let target = env::var("TARGET")?;
    let runtime = library_filename(LIBRARIES[0], &target);
    let verifier = library_filename(LIBRARIES[1], &target);
    let checkout = third_party.join("iree-build-3.14");
    let local = checkout.join("runtime/src/iree/runtime");
    let mirror = env::var_os("IREE_RUNTIME_ASSET_DIR").map(PathBuf::from);
    let (lib_dir, flatcc_dir) = if let Some(explicit) = env::var_os("IREE_RUNTIME_LIB_DIR") {
        let path = PathBuf::from(explicit);
        let flatcc = find_flatcc(&path, &verifier);
        (path, flatcc)
    } else {
        let profile = runtime_assets::profile(&target)?;
        if mirror.is_none()
            && env::var("HOST").as_deref() == Ok(target.as_str())
            && local.join(&runtime).is_file()
            && local_has_drivers(&checkout, profile)
        {
            let flatcc = find_flatcc(&local, &verifier);
            (local, flatcc)
        } else {
            let bundle = runtime_assets::prepare(&target, profile, mirror.as_deref())
                .map_err(|err| format!("{target}/{profile}: {err}"))?;
            (bundle.clone(), bundle)
        }
    };
    let runtime_path = lib_dir.join(runtime);
    let flatcc_path = flatcc_dir.join(verifier);
    for path in [&runtime_path, &flatcc_path] {
        if !path.is_file() {
            return Err(format!("required runtime library missing at {}", path.display()).into());
        }
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for (name, dir) in LIBRARIES.into_iter().zip([&lib_dir, &flatcc_dir]) {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={name}");
    }
    println!("cargo:root={}", lib_dir.display());
    println!("cargo:linked=1");

    // Platform libraries required by the compiled runtime.
    match env::var("CARGO_CFG_TARGET_OS")?.as_str() {
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
        "windows" => {
            println!("cargo:rustc-link-lib=dylib=Synchronization");
        }
        _ => {}
    }
    Ok(())
}

fn check_layout(source: &Path) {
    let include_dir = source.join("runtime/src");
    let header = include_dir.join("iree/runtime/api.h");
    if header.is_file() {
        println!("cargo:rerun-if-changed={}", header.display());
        cc::Build::new()
            .file("csrc/layout_check.c")
            .include(&include_dir)
            .warnings(false)
            .compile("iree_sys_layout_check");
        println!("cargo:rustc-cfg=iree_sys_layout_check");
    }
}

fn local_has_drivers(checkout: &Path, profile: &str) -> bool {
    if profile == "cpu" {
        return true;
    }
    let cache = checkout.join("CMakeCache.txt");
    println!("cargo:rerun-if-changed={}", cache.display());
    fs::read_to_string(cache).is_ok_and(|text| {
        profile.split('-').all(|driver| {
            let setting = format!("IREE_HAL_DRIVER_{}:BOOL=ON", driver.to_ascii_uppercase());
            text.lines().any(|line| line == setting)
        })
    })
}

fn find_flatcc(lib_dir: &Path, filename: &str) -> PathBuf {
    if lib_dir.join(filename).is_file() {
        return lib_dir.to_path_buf();
    }
    // Also accept the legacy just iree layout when using a local build tree.
    lib_dir
        .ancestors()
        .nth(4)
        .map(|build| build.join("build_tools/third_party/flatcc"))
        .filter(|dir| dir.join(filename).is_file())
        .unwrap_or_else(|| lib_dir.to_path_buf())
}
