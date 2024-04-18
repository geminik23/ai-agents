#[cfg(feature = "inbevy")]
mod bevy_main;
#[cfg(feature = "commandline")]
mod cmd_main;

fn main() {
    #[cfg(feature = "commandline")]
    {
        cmd_main::main();
    }

    #[cfg(feature = "inbevy")]
    {
        bevy_main::main();
    }
}
