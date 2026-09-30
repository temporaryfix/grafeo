#![allow(missing_docs)]

fn main() {
    // Header is maintained manually in grafeo.h.
    // cbindgen doesn't support #[unsafe(no_mangle)] (Rust 2024 edition) yet.

    // Installed consumers resolve the shared library through their runtime
    // search path. Cargo's absolute build-directory identity is not portable.
    if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos") {
        println!("cargo:rustc-link-arg-cdylib=-Wl,-install_name,@rpath/libgrafeo_c.dylib");
    }
}
