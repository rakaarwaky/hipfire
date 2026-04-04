use redline::device::Device;
use redline::dispatch::{CommandBuffer, FastDispatch, Kernel};

fn main() {
    let dev = Device::open(None).unwrap();
    let hip_src = "#include <hip/hip_runtime.h>\nextern \"C\" __launch_bounds__(256)\n__global__ void vector_add(const float* a, const float* b, float* c, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) c[i] = a[i] + b[i];\n}\n";
    std::fs::write("/tmp/redline_chain.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3", "-o", "/tmp/redline_chain.hsaco", "/tmp/redline_chain.hip"])
        .output().expect("hipcc");
    assert!(out.status.success());
    let module = dev.load_module_file("/tmp/redline_chain.hsaco").unwrap();
    let kernel = Kernel::find(&module, "vector_add").unwrap();

    let n = 256u32;
    let a = dev.alloc_vram(1024).unwrap();
    let b = dev.alloc_vram(1024).unwrap();
    let c = dev.alloc_vram(1024).unwrap();
    let d = dev.alloc_vram(1024).unwrap();
    let e = dev.alloc_vram(1024).unwrap();
    let fence = dev.alloc_vram(4096).unwrap();
    dev.upload(&a, &f32_bytes(&vec![1.0; 256])).unwrap();
    dev.upload(&b, &f32_bytes(&vec![2.0; 256])).unwrap();
    dev.upload(&c, &vec![0u8; 1024]).unwrap();
    dev.upload(&d, &f32_bytes(&vec![10.0; 256])).unwrap();
    dev.upload(&e, &vec![0u8; 1024]).unwrap();
    dev.upload(&fence, &vec![0u8; 64]).unwrap();

    let fd = FastDispatch::new(&dev, &[&module.code_buf, &a, &b, &c, &d, &e, &fence]).unwrap();

    let mut ka_data = vec![0u8; 1024];
    ka_data[0..8].copy_from_slice(&a.gpu_addr.to_le_bytes());
    ka_data[8..16].copy_from_slice(&b.gpu_addr.to_le_bytes());
    ka_data[16..24].copy_from_slice(&c.gpu_addr.to_le_bytes());
    ka_data[24..28].copy_from_slice(&n.to_le_bytes());
    write_hidden(&mut ka_data, 32, 1, 1, 1, 256, 1, 1);
    ka_data[512..520].copy_from_slice(&c.gpu_addr.to_le_bytes());
    ka_data[520..528].copy_from_slice(&d.gpu_addr.to_le_bytes());
    ka_data[528..536].copy_from_slice(&e.gpu_addr.to_le_bytes());
    ka_data[536..540].copy_from_slice(&n.to_le_bytes());
    write_hidden(&mut ka_data, 544, 1, 1, 1, 256, 1, 1);
    dev.upload(fd.ka_buf_ref(), &ka_data).unwrap();
    let ka_base = fd.ka_buf_ref().gpu_addr;

    // ONLY Test B: Two dispatches WITH barrier (fresh context)
    eprintln!("Two dispatches with CS_PARTIAL_FLUSH barrier...");
    let mut cb = CommandBuffer::new();
    cb.dispatch(kernel, [1, 1, 1], [256, 1, 1], ka_base);
    cb.barrier(fence.gpu_addr, 1);
    cb.dispatch(kernel, [1, 1, 1], [256, 1, 1], ka_base + 512);
    eprintln!("IB: {} dwords", cb.len_dwords());

    match fd.submit_cmdbuf(&dev, &cb) {
        Ok(()) => {
            let c_val = read_f32(&dev, &c);
            let e_val = read_f32(&dev, &e);
            eprintln!("c[0]={} (expect 3), e[0]={} (expect 13)", c_val[0], e_val[0]);
            if (e_val[0] - 13.0).abs() < 0.001 {
                eprintln!("\n=== BARRIER WORKS! ===");
            } else {
                eprintln!("Barrier incomplete: e[0]={} (some elements may be wrong)", e_val[0]);
                let wrong = e_val.iter().filter(|&&v| (v - 13.0).abs() > 0.001).count();
                eprintln!("{}/{} wrong", wrong, n);
            }
        }
        Err(e) => eprintln!("FAILED: {e}"),
    }
    fd.destroy(&dev);
}

fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn write_hidden(d: &mut [u8], off: usize, gx: u32, gy: u32, gz: u32, bx: u16, by: u16, bz: u16) {
    d[off..off+4].copy_from_slice(&gx.to_le_bytes());
    d[off+4..off+8].copy_from_slice(&gy.to_le_bytes());
    d[off+8..off+12].copy_from_slice(&gz.to_le_bytes());
    d[off+12..off+14].copy_from_slice(&bx.to_le_bytes());
    d[off+14..off+16].copy_from_slice(&by.to_le_bytes());
    d[off+16..off+18].copy_from_slice(&bz.to_le_bytes());
}
fn read_f32(dev: &Device, buf: &redline::device::GpuBuffer) -> Vec<f32> {
    let mut r = vec![0u8; buf.size as usize];
    dev.download(buf, &mut r).unwrap();
    r.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}
