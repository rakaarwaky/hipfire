//! Benchmark HFQ4-G128 GEMV vs Q4K at 12288×4096 (Qwen3-8B ffn_gate).
//! Primary question: does HFQ4-G128 hit ≤32 VGPRs?

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    let m = 12288usize;
    let k = 4096usize;

    // HFQ4-G128: 72 bytes per 128 weights
    let groups_per_row = k / 128;
    let row_bytes_hfq4 = groups_per_row * 72;
    let total_hfq4 = m * row_bytes_hfq4;
    let fake_hfq4 = vec![0x55u8; total_hfq4];
    let d_hfq4 = gpu.upload_raw(&fake_hfq4, &[total_hfq4]).unwrap();

    // Q4K: 144 bytes per 256 weights
    let blocks_per_row = k / 256;
    let row_bytes_q4k = blocks_per_row * 144;
    let total_q4k = m * row_bytes_q4k;
    let fake_q4k = vec![0x55u8; total_q4k];
    let d_q4k = gpu.upload_raw(&fake_q4k, &[total_q4k]).unwrap();

    let x_data: Vec<f32> = vec![0.01; k];
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    let n = 100;

    // Warmup both
    gpu.gemv_hfq4g128(&d_hfq4, &d_x, &d_y, m, k).unwrap();
    gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k).unwrap();

    // Benchmark HFQ4-G128
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n {
        gpu.gemv_hfq4g128(&d_hfq4, &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms_hfq4 = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let us_hfq4 = ms_hfq4 * 1000.0 / n as f32;
    let bytes_hfq4 = (total_hfq4 + k * 4) as f64;
    let bw_hfq4 = bytes_hfq4 * n as f64 / (ms_hfq4 as f64 / 1000.0) / 1e9;

    // Benchmark Q4K
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n {
        gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms_q4k = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let us_q4k = ms_q4k * 1000.0 / n as f32;
    let bytes_q4k = (total_q4k + k * 4) as f64;
    let bw_q4k = bytes_q4k * n as f64 / (ms_q4k as f64 / 1000.0) / 1e9;

    eprintln!("=== HFQ4-G128 vs Q4K at {}x{} ===", m, k);
    eprintln!("HFQ4-G128: {:.1} us/call, {:.1} GB/s  ({}B per 128w)", us_hfq4, bw_hfq4, 72);
    eprintln!("Q4K:       {:.1} us/call, {:.1} GB/s  (144B per 256w)", us_q4k, bw_q4k);
    eprintln!("Ratio:     {:.2}x", us_q4k / us_hfq4);

    gpu.free_tensor(d_hfq4).unwrap();
    gpu.free_tensor(d_q4k).unwrap();
    gpu.free_tensor(d_x).unwrap();
    gpu.free_tensor(d_y).unwrap();
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
}
