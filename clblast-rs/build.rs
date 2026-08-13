fn main() {
    let mut config = vcpkg::Config::new();

    if std::env::var("VCPKG_ROOT").is_err() {
        println!("cargo:warning=VCPKG_ROOT is not set. Please set it to the path of your vcpkg installation.");
    }

    unsafe {
        std::env::set_var("VCPKGRS_DYNAMIC", "1"); // DLL 読み込み許可
    }

    let target = std::env::var("TARGET").unwrap_or_default();
    let triplet = match target.as_str() {
        "x86_64-pc-windows-msvc" => "x64-windows",
        "i686-pc-windows-msvc"   => "x86-windows",
        "aarch64-pc-windows-msvc" => "arm64-windows",
        _ => "x64-windows",
    };

    config.target_triplet(triplet);

    if let Err(e) = config.probe("clblast") { // CLBlast 探索
        panic!("Failed to find clblast with vcpkg. Error: {}", e);
    }
}