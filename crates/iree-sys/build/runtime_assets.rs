use std::{
    collections::BTreeMap,
    env, fs, io,
    io::Read,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const REVISION: &str = include_str!("../runtime-revision");
pub const LIBRARIES: [&str; 2] = ["iree_runtime_unified", "flatcc_parsing"];
const RELEASES: &str = "https://github.com/EdwinZhanCN/iree-inference/releases/download";
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    iree_commit: String,
    target: String,
    profile: String,
    files: BTreeMap<String, String>,
}

pub fn profile(target: &str) -> Result<&'static str> {
    let metal = cfg!(feature = "metal");
    let vulkan = cfg!(feature = "vulkan");
    let cuda = cfg!(feature = "cuda");
    match (metal, vulkan, cuda) {
        (false, false, false) => Ok("cpu"),
        (true, false, false) if target == "aarch64-apple-darwin" => Ok("metal"),
        (false, _, _)
            if target.contains("-unknown-linux-gnu") || target.ends_with("-pc-windows-msvc") =>
        {
            Ok(match (vulkan, cuda) {
                (true, true) => "vulkan-cuda",
                (true, false) => "vulkan",
                (false, true) => "cuda",
                (false, false) => unreachable!(),
            })
        }
        _ => Err(format!("unsupported driver features for {target}").into()),
    }
}

pub fn library_filename(name: &str, target: &str) -> String {
    if target.ends_with("-msvc") {
        format!("{name}.lib")
    } else {
        format!("lib{name}.a")
    }
}

pub fn prepare(target: &str, profile: &str, mirror: Option<&Path>) -> Result<PathBuf> {
    let revision = REVISION.trim();
    let stem = format!("iree-runtime-{target}-{profile}");
    let base_url = format!("{RELEASES}/runtime-v1-{revision}-{profile}");
    let cache = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join(&stem);
    fs::create_dir_all(&cache)?;
    let offline = matches!(env::var("CARGO_NET_OFFLINE").as_deref(), Ok("true" | "1"));
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(120)))
        .build()
        .into();

    let manifest_name = format!("{stem}.json");
    let manifest_path = cache.join(&manifest_name);
    let metadata = get(
        &agent,
        &base_url,
        mirror,
        &manifest_path,
        &manifest_name,
        offline,
        32 * 1024,
    )?;
    let manifest: Manifest = serde_json::from_slice(&metadata)?;
    if manifest.schema != 1
        || manifest.iree_commit != revision
        || manifest.target != target
        || manifest.profile != profile
    {
        return Err(format!("incompatible runtime manifest for {target}/{profile}").into());
    }

    for name in LIBRARIES {
        let filename = library_filename(name, target);
        let digest = manifest
            .files
            .get(&filename)
            .ok_or_else(|| format!("missing checksum for {filename}"))?;
        let url_name = format!("{stem}-{filename}");
        let path = cache.join(&filename);
        let mut data = get(
            &agent,
            &base_url,
            mirror,
            &path,
            &url_name,
            offline,
            128 * 1024 * 1024,
        )?;
        if format!("{:x}", Sha256::digest(&data)) != *digest && mirror.is_none() && !offline {
            data = download(&agent, &base_url, &url_name, 128 * 1024 * 1024)?;
        }
        if format!("{:x}", Sha256::digest(&data)) != *digest {
            return Err(format!("SHA-256 mismatch for {url_name}").into());
        }
        cache_file(&path, &data)?;
    }
    cache_file(&manifest_path, &metadata)?;
    Ok(cache)
}

fn get(
    agent: &ureq::Agent,
    url: &str,
    mirror: Option<&Path>,
    cache: &Path,
    name: &str,
    offline: bool,
    limit: u64,
) -> Result<Vec<u8>> {
    if let Some(mirror) = mirror {
        let path = mirror.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        return bounded(fs::File::open(path)?, limit);
    }
    match fs::File::open(cache) {
        Ok(file) => bounded(file, limit),
        Err(err) if err.kind() == io::ErrorKind::NotFound && !offline => {
            download(agent, url, name, limit)
        }
        Err(err) => Err(err.into()),
    }
}

fn download(agent: &ureq::Agent, base: &str, name: &str, limit: u64) -> Result<Vec<u8>> {
    let url = format!("{base}/{name}");
    let response = agent
        .get(&url)
        .call()
        .map_err(|err| format!("{url}: {err}"))?;
    bounded(response.into_body().into_reader(), limit)
}

fn bounded(reader: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("runtime asset exceeds the size limit".into());
    }
    Ok(bytes)
}

fn cache_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if fs::read(path).as_deref().is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    let staged = path.with_extension("download");
    fs::write(&staged, bytes)?;
    fs::rename(staged, path)
}
