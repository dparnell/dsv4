//! Resumable, parallel Hugging Face downloader.
//!
//! Each file is fetched as a set of fixed-size byte ranges by a pool of concurrent
//! connections, written in place into a preallocated `.part` file. Finished chunks are
//! recorded in a sidecar `.part.json`, so an interrupted download resumes where it stopped.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

pub const DEFAULT_REPO: &str = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF";
/// The GGUF carries its own vocab, but the exact HF tokenizer (normalizers + pre-tokenizer)
/// comes from the base repo.
pub const BASE_REPO: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
const CHUNK: u64 = 64 << 20;

#[derive(Debug, Clone, Deserialize)]
pub struct RepoEntry {
    #[serde(rename = "type")]
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct PartState {
    size: u64,
    chunk: u64,
    done: BTreeSet<u64>,
}

pub struct Downloader {
    client: reqwest::Client,
    token: Option<String>,
    connections: usize,
}

fn write_at(f: &File, buf: &[u8], mut off: u64) -> std::io::Result<()> {
    let mut b = buf;
    while !b.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_write(f, b, off)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::write_at(f, b, off)?;
        b = &b[n..];
        off += n as u64;
    }
    Ok(())
}

impl Downloader {
    pub fn new(connections: usize) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent("dsv4-downloader/0.1")
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(120))
            .build()?;
        let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
        Ok(Self { client, token, connections: connections.max(1) })
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let r = self.client.get(url);
        match &self.token {
            Some(t) => r.bearer_auth(t),
            None => r,
        }
    }

    pub async fn list(&self, repo: &str) -> Result<Vec<RepoEntry>> {
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
        let entries: Vec<RepoEntry> = self.get(&url).send().await?.error_for_status()?.json().await?;
        Ok(entries.into_iter().filter(|e| e.kind == "file").collect())
    }

    /// Download `files` (repo-relative paths with sizes) from `repo` into `dir`, keeping paths.
    pub async fn fetch(&self, repo: &str, files: &[RepoEntry], dir: &Path) -> Result<()> {
        let total: u64 = files.iter().map(|f| f.size).sum();
        let mp = MultiProgress::new();
        let overall = mp.add(ProgressBar::new(total));
        overall.set_style(
            ProgressStyle::with_template(
                "{prefix:>8} [{elapsed_precise}] {wide_bar:.cyan/blue} {binary_bytes}/{binary_total_bytes} {binary_bytes_per_sec} eta {eta}",
            )?
            .progress_chars("=> "),
        );
        overall.set_prefix("total");
        for f in files {
            let dest = dir.join(&f.path);
            self.fetch_one(repo, f, &dest, &mp, &overall).await.with_context(|| format!("downloading {}", f.path))?;
        }
        overall.finish();
        Ok(())
    }

    async fn fetch_one(&self, repo: &str, entry: &RepoEntry, dest: &Path, mp: &MultiProgress, overall: &ProgressBar) -> Result<()> {
        let part = PathBuf::from(format!("{}.part", dest.display()));
        let state_path = PathBuf::from(format!("{}.part.json", dest.display()));
        if let Ok(md) = std::fs::metadata(dest) {
            if md.len() == entry.size && !state_path.exists() {
                overall.inc(entry.size);
                return Ok(());
            }
        }
        if let Some(p) = dest.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut state: PartState = std::fs::read(&state_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .filter(|s: &PartState| s.size == entry.size && s.chunk == CHUNK && part.exists())
            .unwrap_or_else(|| PartState { size: entry.size, chunk: CHUNK, done: BTreeSet::new() });
        let file = OpenOptions::new().create(true).write(true).truncate(false).open(&part)?;
        if file.metadata()?.len() != entry.size {
            file.set_len(entry.size)?;
        }
        let file = Arc::new(file);
        let n_chunks = entry.size.div_ceil(CHUNK).max(1);
        let already: u64 = state.done.iter().map(|&c| (entry.size - c * CHUNK).min(CHUNK)).sum();
        overall.inc(already);

        let pb = mp.add(ProgressBar::new(entry.size));
        pb.set_style(ProgressStyle::with_template("{msg:>40} {bar:30} {binary_bytes}/{binary_total_bytes} {binary_bytes_per_sec}")?);
        pb.set_message(entry.path.rsplit('/').next().unwrap_or(&entry.path).to_string());
        pb.set_position(already);

        let url = format!("https://huggingface.co/{repo}/resolve/main/{}", entry.path);
        let todo: Vec<u64> = (0..n_chunks).filter(|c| !state.done.contains(c)).collect();
        let state_m = Arc::new(Mutex::new(std::mem::take(&mut state)));
        let results = futures::stream::iter(todo.into_iter().map(|c| {
            let file = file.clone();
            let url = url.clone();
            let pb = pb.clone();
            let overall = overall.clone();
            let state_m = state_m.clone();
            let state_path = state_path.clone();
            let size = entry.size;
            async move {
                let start = c * CHUNK;
                let end = (start + CHUNK).min(size) - 1;
                let mut attempt = 0u32;
                loop {
                    match self.fetch_range(&url, &file, start, end, &pb, &overall).await {
                        Ok(()) => break,
                        Err(e) => {
                            attempt += 1;
                            if attempt > 12 {
                                return Err(e);
                            }
                            pb.println(format!("chunk {c} of {url}: {e:#} (retry {attempt})"));
                            tokio::time::sleep(Duration::from_secs((1u64 << attempt.min(6)).min(60))).await;
                        }
                    }
                }
                let snapshot = {
                    let mut s = state_m.lock();
                    s.done.insert(c);
                    serde_json::to_vec(&*s)?
                };
                std::fs::write(&state_path, snapshot)?;
                anyhow::Ok(())
            }
        }))
        .buffer_unordered(self.connections)
        .collect::<Vec<_>>()
        .await;
        for r in results {
            r?;
        }
        file.sync_all()?;
        drop(file);
        std::fs::rename(&part, dest)?;
        let _ = std::fs::remove_file(&state_path);
        pb.finish_and_clear();
        Ok(())
    }

    async fn fetch_range(&self, url: &str, file: &File, start: u64, end: u64, pb: &ProgressBar, overall: &ProgressBar) -> Result<()> {
        let resp = self.get(url).header("Range", format!("bytes={start}-{end}")).send().await?;
        let status = resp.status();
        if status != reqwest::StatusCode::PARTIAL_CONTENT && !(status.is_success() && start == 0) {
            bail!("unexpected HTTP status {status}");
        }
        let mut off = start;
        let mut stream = resp.bytes_stream();
        let mut written = 0u64;
        let result: Result<()> = async {
            while let Some(bytes) = stream.next().await {
                let bytes = bytes?;
                let take = (bytes.len() as u64).min(end + 1 - off) as usize;
                write_at(file, &bytes[..take], off)?;
                off += take as u64;
                written += take as u64;
                pb.inc(take as u64);
                overall.inc(take as u64);
                if off > end {
                    break;
                }
            }
            if off != end + 1 {
                bail!("short read: got {} of {} bytes", off - start, end + 1 - start);
            }
            Ok(())
        }
        .await;
        if result.is_err() {
            // roll the bars back: the whole chunk is retried
            pb.set_position(pb.position().saturating_sub(written));
            overall.set_position(overall.position().saturating_sub(written));
        }
        result
    }
}

/// `-00002-of-00006` at the end of a file name without extension: the rest of it, and the count.
fn split_shard(stem: &str) -> Option<(&str, usize)> {
    let (rest, count) = stem.rsplit_once("-of-")?;
    let (name, index) = rest.rsplit_once('-')?;
    index.parse::<usize>().ok()?;
    Some((name, count.parse().ok()?))
}

/// The quant a GGUF of the repo belongs to: its folder, or its name up to the shard number.
fn quant_of(path: &str) -> Option<&str> {
    let stem = path.strip_suffix(".gguf")?;
    Some(match path.split_once('/') {
        Some((folder, _)) => folder,
        None => split_shard(stem).map_or(stem, |s| s.0),
    })
}

/// Files that make up one quant: its shard folder, or the files named so (the end of the name is
/// enough). An Engram table kept in a file of its own comes along if a shard is missing.
pub fn select_quant(entries: &[RepoEntry], quant: &str) -> Result<Vec<RepoEntry>> {
    let ggufs = || entries.iter().filter(|e| e.path.ends_with(".gguf"));
    let named = |e: &&RepoEntry| quant_of(&e.path) == Some(quant);
    let mut files: Vec<RepoEntry> = ggufs().filter(named).cloned().collect();
    if files.is_empty() {
        let end = format!("-{}", quant.to_lowercase());
        files = ggufs().filter(|e| !e.path.contains('/') && quant_of(&e.path).is_some_and(|q| q.to_lowercase().ends_with(&end))).cloned().collect();
        let mut names: Vec<&str> = files.iter().filter_map(|f| quant_of(&f.path)).collect();
        names.sort_unstable();
        names.dedup();
        if names.len() > 1 {
            bail!("several quants end in {quant}: {}", names.join(", "));
        }
    }
    let shards = files.iter().find_map(|f| split_shard(f.path.strip_suffix(".gguf")?)).map_or(files.len(), |s| s.1);
    if files.len() < shards {
        let engram = |e: &&RepoEntry| e.path.to_lowercase().contains("engram") && !files.iter().any(|f| f.path == e.path);
        match ggufs().filter(engram).min_by_key(|e| &e.path) {
            Some(e) if files.len() + 1 == shards => files.push(e.clone()),
            _ => bail!("{quant} has {} of {shards} shards in the repo", files.len()),
        }
    }
    Ok(files)
}

pub async fn run(repo: &str, quant: &str, dir: &Path, connections: usize, list_only: bool) -> Result<()> {
    let dl = Downloader::new(connections)?;
    let entries = dl.list(repo).await?;
    if list_only {
        let mut quants: Vec<(String, u64)> = Vec::new();
        for e in &entries {
            if let Some(q) = quant_of(&e.path) {
                match quants.iter_mut().find(|(n, _)| n == q) {
                    Some((_, s)) => *s += e.size,
                    None => quants.push((q.to_string(), e.size)),
                }
            }
        }
        println!("{repo}:");
        for (q, s) in quants {
            println!("  {q:<14} {:>8.1} GB", s as f64 / 1e9);
        }
        return Ok(());
    }
    let mut files = select_quant(&entries, quant)?;
    // metadata + dense shards first, the big expert / engram shards last
    files.sort_by_key(|f| f.size);
    if files.is_empty() {
        bail!("no GGUF shards for quant {quant} in {repo} (try --list)");
    }
    let total: u64 = files.iter().map(|f| f.size).sum();
    println!("{} shard(s), {:.1} GB -> {}", files.len(), total as f64 / 1e9, dir.display());
    std::fs::create_dir_all(dir)?;
    // tokenizer from the base repo (small)
    let base = dl.list(BASE_REPO).await.unwrap_or_default();
    let tok: Vec<RepoEntry> =
        base.into_iter().filter(|e| e.path == "tokenizer.json" || e.path == "tokenizer_config.json").collect();
    if !tok.is_empty() {
        dl.fetch(BASE_REPO, &tok, dir).await?;
    }
    dl.fetch(repo, &files, dir).await?;
    println!("done.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_quants_in_folders_and_flat() {
        let entries: Vec<RepoEntry> = [
            "UD-IQ1_S/m-00001-of-00002.gguf",
            "UD-IQ1_S/m-00002-of-00002.gguf",
            "m-IQ2_XS-00001-of-00003.gguf",
            "m-IQ2_XS-00003-of-00003.gguf",
            "m-Engram-IQ4_NL.gguf",
            "m-Q2.gguf",
            "b-IQ3_S-00001-of-00009.gguf",
        ]
        .map(|p| RepoEntry { kind: "file".into(), path: p.into(), size: 1 })
        .into();
        let names = |q: &str| select_quant(&entries, q).map(|v| v.into_iter().map(|e| e.path).collect::<Vec<_>>());
        assert_eq!(names("UD-IQ1_S").unwrap().len(), 2);
        assert_eq!(names("iq2_xs").unwrap(), ["m-IQ2_XS-00001-of-00003.gguf", "m-IQ2_XS-00003-of-00003.gguf", "m-Engram-IQ4_NL.gguf"]);
        assert_eq!(names("m-Q2").unwrap(), ["m-Q2.gguf"]);
        assert!(names("IQ3_S").is_err());
        assert!(names("nope").unwrap().is_empty());
    }
}
