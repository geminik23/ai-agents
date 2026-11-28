fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if matches!(
        (
            std::env::var("CARGO_CFG_TARGET_OS").as_deref(),
            std::env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
            std::env::var("CARGO_CFG_TARGET_ENV").as_deref(),
        ),
        (Ok("windows"), Ok("x86_64"), Ok("msvc"))
    ) {
        // The main thread hosts CLI parsing and foreground task execution.
        // Native debug subprocess tests overflowed with a 2 MiB reserve and passed with 4 MiB.
        const STACK_RESERVE_BYTES: u32 = 4 * 1024 * 1024;
        println!("cargo:rustc-link-arg-bin=ai-agents-cli=/STACK:{STACK_RESERVE_BYTES}");
    }
}
