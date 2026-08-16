fn main() {
    if let Err(error) = kcastle_desktop::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
