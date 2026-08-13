fn main() {
    let mut config = vcpkg::Config::new();

    unsafe { 
        if std::env::var("VCPKG_ROOT").is_err() {
            panic!("VCPKG_ROOT is not set.");
        }

        std::env::set_var("VCPKGRS_DYNAMIC", "1"); // DLL 読み込み許可
    };
    config.target_triplet("x64-windows");
    
    if let Err(e) = config.probe("clblast") { // CLBlast 探索
        panic!("Failed to find clblast with vcpkg. Error: {}", e);
    }
}