// #[cfg_attr(feature = "inbevy", derive(Resource))]
// #[cfg(feature="inbevy")]

#[cfg(feature = "commandline")]
mod cmd_main;
#[cfg(feature = "commandline")]
fn main() {
    cmd_main::main()
}

#[cfg(feature = "inbevy")]
mod bevy_main;

#[cfg(feature = "inbevy")]
fn main() {
    bevy_main::main()
}
