fn main() {
    let size = genesis::GenesisCoreState::SIZE;
    println!(
        "GenesisCoreState: {} bytes ({:.2} KB)",
        size,
        size as f64 / 1024.0
    );
    println!("Fits in 4KB page: {}", size <= 4096);
}
