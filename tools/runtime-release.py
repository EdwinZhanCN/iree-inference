"""Build, package, or smoke-test native IREE runtime release assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REVISION = (ROOT / "crates/iree-sys/runtime-revision").read_text().strip()
PROFILES = ("cpu", "metal", "vulkan", "cuda", "vulkan-cuda")

RUNNERS = {
    "aarch64-apple-darwin": "macos-15",
    "x86_64-apple-darwin": "macos-15-intel",
    "aarch64-unknown-linux-gnu": "ubuntu-22.04-arm",
    "x86_64-unknown-linux-gnu": "ubuntu-22.04",
    "aarch64-pc-windows-msvc": "windows-11-arm",
    "x86_64-pc-windows-msvc": "windows-2022",
}

SMOKE = """
use iree_runtime::{BufferView, Instance, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = Instance::new()?;
    let mut args = std::env::args_os().skip(1);
    let device = instance.create_device(args.next().unwrap().to_str().unwrap())?;
    let session = Session::from_file(&device, args.next().unwrap())?;
    let input = BufferView::from_slice(&device, &[4], &[-4.0_f32, -2.0, 0.0, 1.0])?;
    let output = session.call("basic.abs")?.invoke_with(&[&input])?;
    assert_eq!(output[0].to_vec::<f32>()?, [4.0, 2.0, 0.0, 1.0]);
    Ok(())
}
"""


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def supports(target, profile):
    if profile == "metal":
        return target == "aarch64-apple-darwin"
    if profile == "cpu":
        return True
    return "apple" not in target


def cuda_deps(directory):
    """Fetches the CUDA files that the runtime build needs on Windows ARM64.

    The pinned toolkit fetcher has no Windows ARM64 package. The runtime needs
    only headers and libdevice, which do not depend on the platform, so they
    come from the Windows x64 package of the same CUDA version. The runtime
    loads nvcuda.dll when it runs and links nothing from the toolkit.
    """
    sdk = ROOT / "third_party/iree-3.14/third_party/nvidia_sdk_download"
    version = re.search(r'^VERSION = "([^"]+)"', (sdk / "fetch_cuda_toolkit.py").read_text(),
                        re.MULTILINE).group(1)
    directory.mkdir(parents=True, exist_ok=True)
    # The same components as the fetcher: cuda_fp16.hpp includes CCCL's nv/target.
    for component in ("cuda_cccl", "cuda_cudart", "cuda_nvcc"):
        run(sys.executable, sdk / "parse_redist.py", "--label", version, "--product", "cuda",
            "--os", "windows", "--arch", "x86_64", "--component", component,
            "--output", directory, cwd=directory)
    root = directory / "windows-x86_64"
    for required in ("include/cuda.h", "include/nv/target", "nvvm/libdevice/libdevice.10.bc"):
        if not (root / required).is_file():
            raise SystemExit(f"the CUDA {version} package has no {required}")
    return root


def native_build(profile, target):
    run("just", "_iree-clone", cwd=ROOT)
    if "cuda" in profile and target == "aarch64-pc-windows-msvc":
        os.environ["IREE_CUDA_DEPS_DIR"] = str(cuda_deps(ROOT / "third_party/cuda-deps"))
    if "vulkan" in profile:
        run("git", "-C", ROOT / "third_party/iree-3.14", "submodule", "update",
            "--init", "--depth=1", "third_party/vulkan_headers")
    opts = [
        f"-DIREE_HAL_DRIVER_{driver.upper()}={'ON' if driver in profile else 'OFF'}"
        for driver in ("metal", "vulkan", "cuda")
    ]
    if target == "aarch64-pc-windows-msvc":
        elf_cmake = ROOT / "third_party/iree-3.14/runtime/src/iree/hal/local/elf/CMakeLists.txt"
        guard = 'if(IREE_ARCH STREQUAL "x86_64")'
        source = elf_cmake.read_text(encoding="utf-8")
        if guard not in source:
            # The fork's 64-bit check also matches ARM64 and adds an x64 object.
            old_guard = "if(MSVC_C_ARCHITECTURE_ID MATCHES 64 OR MSVC_CXX_ARCHITECTURE_ID MATCHES 64)"
            if source.count(old_guard) != 1:
                raise SystemExit("IREE's embedded ELF MASM guard changed")
            elf_cmake.write_text(source.replace(old_guard, guard), encoding="utf-8")
        # The baseline NEON kernels use Clang vector and FP16 intrinsics.
        # Keep the MSVC ABI while compiling them with Clang's CL frontend.
        opts.extend(
            f"-DCMAKE_{language}_COMPILER{suffix}={value}"
            for language in ("C", "CXX")
            for suffix, value in (("", "clang-cl"), ("_TARGET", target))
        )
        # The fork has no clang-cl feature flags for the optional Arm variants.
        opts.extend(
            f"-DIREE_UK_BUILD_ARM_64_{isa}:BOOL=OFF"
            for isa in ("FULLFP16", "FP16FML", "BF16", "DOTPROD", "I8MM")
        )
    run("just", "iree", *opts, cwd=ROOT)


def package(target, profile, output, build):
    pinned = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["metadata"]["iree"]["commit"]
    if pinned != REVISION:
        raise SystemExit("the runtime release revision must match Cargo workspace metadata")
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    source = next(x.split("=", 1)[1] for x in cache if x.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="))
    actual = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise SystemExit(f"native build uses IREE {actual}; expected {REVISION}")
    for driver in ("metal", "vulkan", "cuda", "local_sync", "local_task"):
        enabled = driver in profile or driver.startswith("local_")
        option = f"IREE_HAL_DRIVER_{driver.upper()}:BOOL={'ON' if enabled else 'OFF'}"
        if option not in cache:
            raise SystemExit(f"native build must have {option}")
    stem = f"iree-runtime-{target}-{profile}"
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for library, directory in (
        ("iree_runtime_unified", "runtime/src/iree/runtime"),
        ("flatcc_parsing", "build_tools/third_party/flatcc"),
    ):
        filename = f"{library}.lib" if target.endswith("-msvc") else f"lib{library}.a"
        path = output / f"{stem}-{filename}"
        archive = build / directory / filename
        if library == "iree_runtime_unified" and target == "x86_64-pc-windows-msvc":
            # The fork adds the MASM thunk to the ELF arch static target.
            # The unified runtime aggregates object targets and omits it.
            run("cmake", "--build", build, "--target", "iree_hal_local_elf_arch")
            thunk = build / "runtime/src/iree/hal/local/elf/arch/x86_64_msvc.obj"
            run("lib.exe", "/NOLOGO", f"/OUT:{path.resolve()}", archive, thunk)
        else:
            shutil.copyfile(archive, path)
        with path.open("rb") as stream:
            files[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    licenses = (source, Path(source) / "third_party/flatcc")
    (output / f"{stem}-licenses.txt").write_text("\n\n".join(
        (Path(path) / "LICENSE").read_text(encoding="utf-8") for path in licenses
    ), encoding="utf-8")
    manifest = dict(schema=1, iree_commit=REVISION, target=target, profile=profile, files=files)
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def smoke(profile, assets, vmfb, driver):
    with tempfile.TemporaryDirectory(prefix="iree-runtime-smoke-") as directory:
        stage = Path(directory)
        dependency = stage / "dependency"
        dependency.mkdir()
        shutil.copyfile(ROOT / "Cargo.toml", dependency / "Cargo.toml")
        for crate in ("iree-runtime", "iree-sys"):
            shutil.copytree(ROOT / "crates" / crate, dependency / "crates" / crate)
        run("git", "init", "-q", dependency)
        run("git", "-C", dependency, "add", ".")
        run("git", "-C", dependency, "-c", "commit.gpgsign=false", "-c", "user.name=Smoke test", "-c",
            "user.email=smoke@example.invalid", "commit", "-qm", "Package smoke fixture")
        revision = subprocess.check_output(["git", "-C", dependency, "rev-parse", "HEAD"], text=True).strip()
        app = stage / "app"
        (app / "src").mkdir(parents=True)
        features = [] if profile == "cpu" else profile.split("-")
        (app / "Cargo.toml").write_text(f'''[package]
name = "iree-runtime-consumer"
version = "0.0.0"
edition = "2024"

[dependencies]
iree-runtime = {{ git = "{dependency.as_uri()}", rev = "{revision}", features = {json.dumps(features)} }}
''')
        (app / "src/main.rs").write_text(SMOKE)
        env = os.environ.copy()
        for name in ("IREE_RUNTIME_LIB_DIR", "IREE_SOURCE_DIR", "CARGO_TARGET_DIR"):
            env.pop(name, None)
        env["IREE_RUNTIME_ASSET_DIR"] = str(assets.resolve())
        # Build as an independent consumer with no original checkout or native build tree.
        run("cargo", "+stable", "run", "--release", "--manifest-path", app / "Cargo.toml",
            "--", driver, vmfb.resolve(), env=env, cwd=app)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="command", required=True)
    matrix = subs.add_parser("matrix")
    pack = subs.add_parser("package")
    test = subs.add_parser("smoke")
    for sub in (matrix, pack, test):
        sub.add_argument("profile", choices=PROFILES)
    matrix.add_argument("--target", default="", help="build only this target")
    pack.add_argument("target", choices=RUNNERS)
    pack.add_argument("output", type=Path)
    pack.add_argument("--build", type=Path, default=ROOT / "third_party/iree-build-3.14")
    pack.add_argument("--package-only", action="store_true")
    test.add_argument("assets", type=Path)
    test.add_argument("vmfb", type=Path)
    test.add_argument("--driver", default="local-task")
    args = cli.parse_args()
    if args.command == "matrix":
        print(json.dumps({"include": [
            {"target": target, "runner": runner, "arch": "arm64" if target.startswith("aarch64") else "x64"}
            for target, runner in RUNNERS.items()
            if supports(target, args.profile) and args.target in ("", target)
        ]}))
        return
    if args.command == "smoke":
        smoke(args.profile, args.assets, args.vmfb, args.driver)
        return
    if not supports(args.target, args.profile):
        cli.error(f"{args.profile} is unavailable for {args.target}")
    host = next(line.split(": ", 1)[1] for line in
                subprocess.check_output(["rustc", "+stable", "-vV"], text=True).splitlines()
                if line.startswith("host: "))
    if host != args.target:
        cli.error(f"native packaging must run on {args.target}, found {host}")
    if not args.package_only:
        native_build(args.profile, args.target)
    package(args.target, args.profile, args.output, args.build)


if __name__ == "__main__":
    main()
