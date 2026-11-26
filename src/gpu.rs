//! Thin wgpu layer: device setup, buffers with stable ids, a compute-pipeline registry,
//! a bind-group cache and a command batch that records dispatches with immediates.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;

use crate::stats::Stats;

pub struct Buf {
    pub buf: wgpu::Buffer,
    pub id: u64,
    pub size: u64,
    gpu_bytes: Arc<AtomicU64>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        self.gpu_bytes.fetch_sub(self.size, Ordering::Relaxed);
    }
}

/// A (sub-)range of a buffer to bind.
#[derive(Clone, Copy)]
pub struct B<'a> {
    pub buf: &'a Buf,
    pub offset: u64,
    pub size: u64,
}

impl Buf {
    pub fn all(&self) -> B<'_> {
        B { buf: self, offset: 0, size: self.size }
    }
    pub fn range(&self, offset: u64, size: u64) -> B<'_> {
        B { buf: self, offset, size }
    }
}

impl<'a> From<&'a Buf> for B<'a> {
    fn from(b: &'a Buf) -> Self {
        b.all()
    }
}

struct Kernel {
    id: u64,
    src: String,
    pipe: OnceLock<wgpu::ComputePipeline>,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pipelines: HashMap<&'static str, Kernel>,
    bg_cache: Mutex<HashMap<(u64, Vec<(u64, u64, u64)>), wgpu::BindGroup>>,
    next_id: AtomicU64,
    pub gpu_bytes: Arc<AtomicU64>,
    pub stats: Arc<Stats>,
    readback: Mutex<Option<wgpu::Buffer>>,
    pub prof: Option<Prof>,
}

/// With DSV4_PROFILE set: GPU time per kernel while decoding, from timestamps around each dispatch
/// (which then runs in a pass of its own, and each submit waits for the GPU).
pub struct Prof {
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period: f32,
    pub on: std::sync::atomic::AtomicBool,
}
const PROF_N: u32 = 2048;

impl Gpu {
    pub fn new(stats: Arc<Stats>, backend: Option<&str>) -> Result<Self> {
        let backends = match backend.map(|s| s.to_ascii_lowercase()) {
            Some(ref s) if s == "vulkan" => wgpu::Backends::VULKAN,
            Some(ref s) if s == "dx12" => wgpu::Backends::DX12,
            Some(ref s) if s == "metal" => wgpu::Backends::METAL,
            _ => wgpu::Backends::PRIMARY,
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(backends));
        // prefer a discrete GPU, then Vulkan over DX12
        let adapter = adapters
            .into_iter()
            .filter(|a| {
                let f = a.features();
                f.contains(wgpu::Features::IMMEDIATES) && f.contains(wgpu::Features::SUBGROUP)
            })
            .max_by_key(|a| {
                let i = a.get_info();
                let ty = match i.device_type {
                    wgpu::DeviceType::DiscreteGpu => 3,
                    wgpu::DeviceType::IntegratedGpu => 2,
                    _ => 0,
                };
                let be = match i.backend {
                    wgpu::Backend::Vulkan => 2,
                    wgpu::Backend::Metal => 2,
                    wgpu::Backend::Dx12 => 1,
                    _ => 0,
                };
                (ty, be, a.limits().max_buffer_size)
            })
            .context("no GPU adapter with IMMEDIATES + SUBGROUP support found")?;
        let info = adapter.get_info();
        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage.min(16);
        limits.max_compute_workgroup_storage_size = al.max_compute_workgroup_storage_size;
        limits.max_compute_invocations_per_workgroup = al.max_compute_invocations_per_workgroup.min(1024);
        limits.max_compute_workgroup_size_x = al.max_compute_workgroup_size_x.min(1024);
        limits.max_compute_workgroups_per_dimension = al.max_compute_workgroups_per_dimension;
        limits.max_immediate_size = al.max_immediate_size.min(128);
        if limits.max_immediate_size < 64 {
            bail!("adapter immediate size too small ({})", al.max_immediate_size);
        }
        let profile = std::env::var_os("DSV4_PROFILE").is_some() && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("dsv4"),
            required_features: wgpu::Features::IMMEDIATES
                | wgpu::Features::SUBGROUP
                | if profile { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))?;
        device.on_uncaptured_error(Arc::new(|e| {
            eprintln!("wgpu error: {e}");
            std::process::abort();
        }));
        if profile {
            stats.profiling.store(true, Ordering::Relaxed);
            tracing::warn!("DSV4_PROFILE is set: timing every GPU kernel, decoding will be much slower");
        } else if std::env::var_os("DSV4_PROFILE").is_some() {
            tracing::warn!("DSV4_PROFILE is set but this GPU has no timestamp queries; not profiling");
        }
        let prof = profile.then(|| Prof {
            qs: device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: PROF_N * 2 }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: PROF_N as u64 * 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: PROF_N as u64 * 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period: queue.get_timestamp_period(),
            on: std::sync::atomic::AtomicBool::new(false),
        });
        let mut g = Self {
            prof,
            device,
            queue,
            info,
            limits,
            pipelines: HashMap::new(),
            bg_cache: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            gpu_bytes: Arc::new(AtomicU64::new(0)),
            stats,
            readback: Mutex::new(None),
        };
        g.build_pipelines()?;
        Ok(g)
    }

    fn build_pipelines(&mut self) -> Result<()> {
        for (name, src) in crate::kernels::all().into_iter().chain(crate::kernels::on_demand()) {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.pipelines.insert(name, Kernel { id, src, pipe: OnceLock::new() });
        }
        crate::kernels::all().iter().for_each(|k| self.prepare(k.0));
        Ok(())
    }

    /// Build a kernel now rather than when it is first run.
    pub fn prepare(&self, kernel: &'static str) {
        self.pipeline(kernel);
    }

    fn pipeline(&self, name: &'static str) -> (u64, &wgpu::ComputePipeline) {
        let k = &self.pipelines[name];
        let pipe = k.pipe.get_or_init(|| {
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(k.src.as_str().into()),
            });
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        (k.id, pipe)
    }

    pub fn buffer(&self, label: &str, size: u64) -> Buf {
        let size = size.max(16).div_ceil(16) * 16;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.gpu_bytes.fetch_add(size, Ordering::Relaxed);
        Buf { buf, id: self.next_id.fetch_add(1, Ordering::Relaxed), size, gpu_bytes: self.gpu_bytes.clone() }
    }

    pub fn buffer_init(&self, label: &str, data: &[u8]) -> Buf {
        let b = self.buffer(label, data.len() as u64);
        self.write(&b, 0, data);
        b
    }

    pub fn readback_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.div_ceil(16) * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn write(&self, b: &Buf, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        // write_buffer needs a 4-byte multiple
        if data.len() % 4 != 0 {
            let mut v = data.to_vec();
            v.resize(data.len().div_ceil(4) * 4, 0);
            self.queue.write_buffer(&b.buf, offset, &v);
        } else {
            self.queue.write_buffer(&b.buf, offset, data);
        }
        self.stats.upload_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
    }

    pub fn bind_group(&self, kernel: &'static str, entries: &[B<'_>]) -> wgpu::BindGroup {
        let (pid, pipe) = self.pipeline(kernel);
        let key = (pid, entries.iter().map(|e| (e.buf.id, e.offset, e.size)).collect::<Vec<_>>());
        if let Some(bg) = self.bg_cache.lock().get(&key) {
            return bg.clone();
        }
        let layout = pipe.get_bind_group_layout(0);
        let e: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &b.buf.buf,
                    offset: b.offset,
                    size: std::num::NonZeroU64::new(b.size),
                }),
            })
            .collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(kernel), layout: &layout, entries: &e });
        let mut c = self.bg_cache.lock();
        if c.len() > 200_000 {
            c.clear();
        }
        c.insert(key, bg.clone());
        bg
    }

    pub fn begin(&self) -> Batch<'_> {
        Batch { gpu: self, enc: Some(self.device.create_command_encoder(&Default::default())), pass: None, n_dispatch: 0, names: Vec::new() }
    }

    /// Push pending buffer writes to the GPU and release their staging memory.
    pub fn flush(&self) {
        self.queue.submit([]);
        self.wait();
    }

    /// Block until all submitted work is done.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    /// Copy `size` bytes from `src` to the host. Submits `batch` first.
    pub fn read(&self, batch: Batch<'_>, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        self.read_multi(batch, &[(src, offset, size)]).pop().unwrap()
    }

    /// Copy several ranges to the host with one submit + one wait.
    pub fn read_multi(&self, mut batch: Batch<'_>, ranges: &[(&Buf, u64, u64)]) -> Vec<Vec<u8>> {
        let total: u64 = ranges.iter().map(|r| r.2.div_ceil(256) * 256).sum();
        let mut cache = self.readback.lock();
        if cache.as_ref().is_none_or(|b| b.size() < total) {
            *cache = Some(self.readback_buffer(total.next_power_of_two().max(1 << 20)));
        }
        let rb = cache.as_ref().unwrap().clone();
        let mut off = 0;
        let mut offs = Vec::new();
        for &(src, so, size) in ranges {
            batch.copy(src, so, &rb, off, size.div_ceil(4) * 4);
            offs.push(off);
            off += size.div_ceil(256) * 256;
        }
        batch.submit();
        let slice = rb.slice(..total);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait();
        rx.recv().expect("map callback").expect("map failed");
        let out = {
            let view = slice.get_mapped_range().expect("mapped");
            ranges.iter().zip(&offs).map(|(r, &o)| view[o as usize..(o + r.2) as usize].to_vec()).collect()
        };
        rb.unmap();
        self.stats.download_bytes.fetch_add(total, Ordering::Relaxed);
        out
    }

    pub fn read_f32(&self, batch: Batch<'_>, src: &Buf, offset_elems: u64, n: usize) -> Vec<f32> {
        let b = self.read(batch, src, offset_elems * 4, n as u64 * 4);
        bytemuck::cast_slice(&b).to_vec()
    }

    pub fn max_binding(&self) -> u64 {
        self.limits.max_storage_buffer_binding_size.min(self.limits.max_buffer_size)
    }
}

pub struct Batch<'g> {
    gpu: &'g Gpu,
    enc: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    pub n_dispatch: u32,
    names: Vec<&'static str>,
}

impl<'g> Batch<'g> {
    fn pass(&mut self) -> &mut wgpu::ComputePass<'static> {
        if self.pass.is_none() {
            let p = self.enc.as_mut().unwrap().begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            self.pass = Some(p.forget_lifetime());
        }
        self.pass.as_mut().unwrap()
    }

    /// Record one dispatch. `params` is passed as the kernel's immediate block.
    pub fn run(&mut self, kernel: &'static str, bufs: &[B<'_>], params: &[u32], groups: (u32, u32, u32)) {
        if groups.0 == 0 || groups.1 == 0 || groups.2 == 0 {
            return;
        }
        let max = self.gpu.limits.max_compute_workgroups_per_dimension;
        assert!(groups.0 <= max && groups.1 <= max && groups.2 <= max, "{kernel}: dispatch {groups:?} too large");
        let bg = self.gpu.bind_group(kernel, bufs);
        let pipe = self.gpu.pipeline(kernel).1;
        assert!(params.len() <= 16);
        let mut imm = [0u32; 16];
        imm[..params.len()].copy_from_slice(params);
        let timed = self.gpu.prof.as_ref().filter(|p| p.on.load(Ordering::Relaxed) && self.names.len() < PROF_N as usize);
        if let Some(pr) = timed {
            let i = self.names.len() as u32;
            self.pass = None;
            let p = self.enc.as_mut().unwrap().begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &pr.qs,
                    beginning_of_pass_write_index: Some(2 * i),
                    end_of_pass_write_index: Some(2 * i + 1),
                }),
            });
            self.pass = Some(p.forget_lifetime());
            self.names.push(kernel);
        }
        let p = self.pass();
        p.set_pipeline(pipe);
        p.set_bind_group(0, &bg, &[]);
        p.set_immediates(0, bytemuck::cast_slice(&imm[..16]));
        p.dispatch_workgroups(groups.0, groups.1, groups.2);
        self.n_dispatch += 1;
        if timed.is_some() {
            self.pass = None;
        }
    }

    pub fn copy(&mut self, src: &Buf, so: u64, dst: &wgpu::Buffer, d_o: u64, size: u64) {
        self.pass = None;
        self.enc.as_mut().unwrap().copy_buffer_to_buffer(&src.buf, so, dst, d_o, size);
    }

    pub fn submit(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        self.pass = None;
        let Some(mut enc) = self.enc.take() else { return };
        let n = self.names.len() as u64;
        let pr = self.gpu.prof.as_ref().filter(|_| n > 0);
        if let Some(pr) = pr {
            enc.resolve_query_set(&pr.qs, 0..2 * n as u32, &pr.resolve, 0);
            enc.copy_buffer_to_buffer(&pr.resolve, 0, &pr.read, 0, n * 16);
        }
        self.gpu.queue.submit([enc.finish()]);
        if let Some(pr) = pr {
            let slice = pr.read.slice(..n * 16);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.wait();
            {
                let v = slice.get_mapped_range().expect("mapped");
                let t: &[u64] = bytemuck::cast_slice(&v);
                let mut tot = self.gpu.stats.kernel_ns.lock();
                for (i, name) in self.names.iter().enumerate() {
                    let e = tot.entry(name).or_default();
                    e.0 += t[2 * i + 1].saturating_sub(t[2 * i]) as f64 * pr.period as f64;
                    e.1 += 1;
                }
            }
            pr.read.unmap();
        }
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}
