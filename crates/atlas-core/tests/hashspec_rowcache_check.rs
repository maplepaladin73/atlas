// quick check: transliteration of the CUDA hy4_engram_hash_batch kernel vs HashLayout::hash_validated
#[test]
fn cuda_kernel_spec_matches_host_hash() {
    use atlas_core::hy4_reference::engram::{multipliers, HashLayout};
    // synthetic but representative layout: compressed vocab 5000 extended to 6000, pad 0
    let vocab = 6000usize;
    let compressed = 5000usize;
    let mut map: Vec<u32> = {
        // pseudo-random permutation-ish map for compressed region
        let mut v: Vec<u32> = (0..compressed as u32).collect();
        // simple deterministic shuffle
        for i in (1..compressed).rev() {
            let j = (i * 2654435761usize.wrapping_mul(i + 7)) % (i + 1);
            v.swap(i, j);
        }
        v
    };
    map.extend((map.len()..vocab).map(|v| v as u32));
    let mult = multipliers(12345, 3, compressed, 3).unwrap();
    let sizes = vec![97usize, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173];
    let layout = HashLayout { map: map.clone(), pad: 0, heads_per_order: 8, multipliers: mult.clone(), sizes: sizes.clone() };
    layout.validate().unwrap();
    let mut offsets = Vec::new();
    let mut n = 0i64;
    for &s in &sizes { offsets.push(n); n += s as i64; }

    // CUDA kernel transliteration (hy4_engram_hash_batch), one head/row
    let kernel_hash = |token: i64, previous: i64, older: i64, h: usize| -> i32 {
        let heads_per_order = 8usize;
        if token < 0 || token as usize >= vocab || token == 0 { return -1; }
        let tokens = [token, previous, older];
        let mut rolling: u64 = 0;
        let mut blocked = false;
        let order = h / heads_per_order + 2;
        for s in 0..order {
            let id = tokens[s];
            blocked |= id < 0 || id as usize >= vocab;
            let source = if blocked { map[0] } else { map[id as usize] };
            rolling ^= (source as u64).wrapping_mul(mult[s]);
        }
        let rem = (rolling as i64).rem_euclid(sizes[h] as i64);
        (rem + offsets[h]) as i32
    };

    // random-ish token streams incl. dead ids and pad, compare all heads
    let mut seed = 0x12345678u64;
    let mut rng = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    let mut checked = 0u64;
    for _case in 0..2000 {
        let len = (rng() % 5) as usize; // history lengths 0..4 incl shorter than lookback
        let mut hist: Vec<i64> = (0..len).map(|_| (rng() % (vocab as u64 + 50)) as i64 - 25).collect();
        let token = (rng() % (vocab as u64 + 50)) as i64 - 25;
        // context construction exactly as forward.rs: oldest-first suffix of <=2
        let mut ctx: Vec<i64> = Vec::new();
        if hist.len() >= 2 { ctx.push(hist[hist.len() - 2]); }
        if let Some(&p) = hist.last() { ctx.push(p); }
        let host = layout.hash_validated(&ctx, token).unwrap();
        // kernel inputs: previous/older with -1 for missing
        let previous = if hist.is_empty() { -1 } else { hist[hist.len() - 1] };
        let older = if hist.len() < 2 { -1 } else { hist[hist.len() - 2] };
        for h in 0..16 {
            let k = kernel_hash(token, previous, older, h);
            assert_eq!(k, host[h], "mismatch head {h} token {token} hist {hist:?}");
            checked += 1;
        }
        hist.clear();
    }
    println!("hash spec parity: {checked} head-rows exact");
}
