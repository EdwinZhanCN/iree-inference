//! Exposes `cfg(iree_linked)` when iree-sys found and linked the IREE runtime
//! library, so integration tests/examples that execute IREE code can be gated on
//! it instead of failing to link on machines without the runtime build.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(iree_linked)");
    println!("cargo:rerun-if-env-changed=DEP_IREE_RUNTIME_UNIFIED_LINKED");
    if std::env::var("DEP_IREE_RUNTIME_UNIFIED_LINKED").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=iree_linked");
    }
}
