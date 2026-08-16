//! VERIFICATION probe for the device ArgMax kernel (shaders::argmax).
//! Uniform [rows][C] f32 layout, C=8192; each row has its maximum at a known
//! index (first / interior / last / tie). Any mismatch is a kernel bug
//! (addressing, tree-reduce, or push constants).
use onnx_vulkan_core::shaders::argmax::ARGMAX;
use vk_compute::{compile_wgsl, VkContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    println!("device: {}", ctx.device_name);
    let spirv = compile_wgsl(ARGMAX)?;

    const C: u32 = 8192;
    let cases: Vec<(u32, u32)> = vec![
        (8192, 8191), // max at last
        (8192, 0),    // max at first
        (8192, 123),  // max interior
        (8192, 4096), // max exactly at half
        (8192, 42),   // tie at 42 and 100 -> 42 (lowest index wins)
        (8192, 8190), // max second-to-last
    ];
    let rows = cases.len() as u32;

    let mut packed: Vec<f32> = vec![-1000.0; (rows as usize) * (C as usize)];
    for (i, &(_, exp)) in cases.iter().enumerate() {
        // all-negative increasing baseline: -1000 + k*0.001 (max at last by
        // default, so a misplaced max is never masked by the baseline)
        for k in 0..C {
            packed[i as usize * C as usize + k as usize] = -1000.0 + (k as f32) * 0.001;
        }
        packed[i as usize * C as usize + exp as usize] = 1e6;
        if exp == 42 {
            packed[i as usize * C as usize + 100] = 1e6; // tie -> expect 42
        }
    }
    let pbytes: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x = ctx.create_storage_buffer(pbytes.len() as u64)?;
    ctx.upload(&x, &pbytes)?;
    let o = ctx.create_storage_buffer(rows as u64 * 8)?;

    let pipe = ctx.create_pipeline(&spirv, 2, 16)?;
    let mut push = vec![];
    for v in [C, 1u32, rows, rows] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    ctx.stream_dispatch(&pipe, &[&x, &o], &push, [rows, 1, 1])?;
    ctx.flush_wait()?;
    let out = ctx.stream_download(&o, (rows as usize) * 8)?;
    let mut bad = 0;
    for (i, &(_, exp)) in cases.iter().enumerate() {
        let got = i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
        let ok = got as u32 == exp;
        if !ok {
            bad += 1;
        }
        println!(
            "row {i}: c={C} expected={exp} got={got} {}",
            if ok { "OK" } else { "WRONG" }
        );
    }
    if bad == 0 {
        println!("ARGMAX KERNEL VERIFIED ({rows}/{rows} rows correct).");
    } else {
        println!("ARGMAX KERNEL WRONG: {bad}/{rows} rows. DO NOT SHIP.");
    }
    Ok(())
}
