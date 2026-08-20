//! Image preprocessing for the qwen3vl encoder, ported from llama.cpp b10423
//! `tools/mtmd/mtmd-image.cpp` (`mtmd_image_preprocessor_dyn_size` +
//! `img_tool`): smart-resize to a multiple of patch*merge within the pixel
//! budget, aspect-preserving bilinear resize with centered black padding
//! (PAD_CEIL — llama.cpp's default, which qwen3vl does not override), then
//! /255 and mean/std normalization. The output is planar [c][y][x] f32,
//! exactly what `VisionModel::encode` consumes.
//!
//! Every rounding choice below (round-half-away, trunc-to-int in the
//! bilinear, floor of the composite offset) mirrors the C++ so the same
//! image bytes produce the same floats the reference feeds its encoder.

/// A decoded, smart-resized, normalized image ready for the encoder, plus a
/// content hash for prompt-cache identity. `planar` is [c][y][x] f32.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub planar: Vec<f32>,
    pub w: u32,
    pub h: u32,
    /// FNV-1a of the original encoded bytes: two prompts share cached prefix
    /// rows for an image span only when the underlying image bytes matched.
    pub hash: u64,
}

impl PreparedImage {
    /// Merged-grid dimensions: one trunk embedding per 2x2 patch block.
    pub fn grid(&self, align: u32) -> (usize, usize) {
        ((self.w / align) as usize, (self.h / align) as usize)
    }

    /// Rows this image occupies in the trunk (= embeddings the encoder emits).
    pub fn n_tokens(&self, align: u32) -> usize {
        let (nx, ny) = self.grid(align);
        nx * ny
    }
}

/// FNV-1a, for image identity in the prompt cache.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Pixel-budget policy, mirroring clip.cpp's `set_limit_image_tokens`.
/// One output token covers `patch*merge` square pixels (32x32 = 1024 px).
#[derive(Debug, Clone)]
pub struct Preprocessor {
    /// patch_size * spatial_merge (32) — every output side aligns to this
    pub align: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Preprocessor {
    /// llama.cpp defaults for qwen3vl: 8..4096 tokens. Its own loader warns
    /// that Qwen-VL grounding wants >= 1024 min tokens; pass `min_tokens`
    /// to match a deployment that sets `--image-min-tokens`.
    pub fn new(hp: &crate::VisionHparams, min_tokens: Option<u32>, max_tokens: Option<u32>) -> Preprocessor {
        let align = (hp.patch * hp.merge) as u32;
        let patch_area = align * align;
        Preprocessor {
            align,
            min_pixels: min_tokens.unwrap_or(8) * patch_area,
            max_pixels: max_tokens.unwrap_or(4096) * patch_area,
            mean: hp.image_mean,
            std: hp.image_std,
        }
    }

    /// `calc_size_preserved_ratio` ("smart_resize"): nearest multiple of
    /// `align` per side, then scaled to fit [min_pixels, max_pixels].
    pub fn target_size(&self, w: u32, h: u32) -> (u32, u32) {
        let f = self.align as f32;
        let round_by = |x: f32| ((x / f).round() as i64 * self.align as i64) as i64;
        let ceil_by = |x: f32| ((x / f).ceil() as i64 * self.align as i64) as i64;
        let floor_by = |x: f32| ((x / f).floor() as i64 * self.align as i64) as i64;

        let mut w_bar = (self.align as i64).max(round_by(w as f32));
        let mut h_bar = (self.align as i64).max(round_by(h as f32));

        if h_bar * w_bar > self.max_pixels as i64 {
            let beta = ((h as f32) * (w as f32) / self.max_pixels as f32).sqrt();
            h_bar = (self.align as i64).max(floor_by(h as f32 / beta));
            w_bar = (self.align as i64).max(floor_by(w as f32 / beta));
        } else if h_bar * w_bar < self.min_pixels as i64 {
            let beta = (self.min_pixels as f32 / ((h as f32) * (w as f32))).sqrt();
            h_bar = ceil_by(h as f32 * beta);
            w_bar = ceil_by(w as f32 * beta);
        }
        (w_bar as u32, h_bar as u32)
    }

    /// Decode + resize + normalize one encoded (JPEG/PNG) image.
    pub fn prepare(&self, bytes: &[u8]) -> Result<PreparedImage, String> {
        let (rgb, w, h) = decode(bytes)?;
        let (planar, tw, th) = self.run(&rgb, w, h)?;
        Ok(PreparedImage { planar, w: tw, h: th, hash: fnv1a(bytes) })
    }

    /// Full pipeline: interleaved RGB bytes -> planar normalized f32 at the
    /// smart-resized dimensions. Returns (planar data, width, height).
    pub fn run(&self, rgb: &[u8], w: u32, h: u32) -> Result<(Vec<f32>, u32, u32), String> {
        if w == 0 || h == 0 || rgb.len() != (w * h * 3) as usize {
            return Err(format!("bad image buffer: {}x{} with {} bytes", w, h, rgb.len()));
        }
        let (tw, th) = self.target_size(w, h);
        let resized = resize_pad_ceil(rgb, w, h, tw, th);

        // /255, then (v - mean) / std, then interleaved -> planar
        let n = (tw * th) as usize;
        let mut out = vec![0f32; n * 3];
        for c in 0..3 {
            let (m, s) = (self.mean[c], self.std[c]);
            for i in 0..n {
                out[c * n + i] = (resized[i * 3 + c] as f32 / 255.0 - m) / s;
            }
        }
        Ok((out, tw, th))
    }
}

/// `img_tool::resize` with PAD_CEIL: aspect-preserving bilinear resize with
/// ceil'd dimensions, composited centered onto a black target.
fn resize_pad_ceil(src: &[u8], sw: u32, sh: u32, tw: u32, th: u32) -> Vec<u8> {
    if (sw, sh) == (tw, th) {
        return src.to_vec();
    }
    let scale = (tw as f32 / sw as f32).min(th as f32 / sh as f32);
    let nw = ((sw as f32 * scale).ceil() as u32).min(tw).max(1);
    let nh = ((sh as f32 * scale).ceil() as u32).min(th).max(1);

    let resized = resize_bilinear(src, sw, sh, nw, nh);

    let mut dst = vec![0u8; (tw * th * 3) as usize];
    let ox = ((tw - nw) / 2) as usize;
    let oy = ((th - nh) / 2) as usize;
    for y in 0..nh as usize {
        let drow = ((y + oy) * tw as usize + ox) * 3;
        let srow = y * nw as usize * 3;
        dst[drow..drow + nw as usize * 3].copy_from_slice(&resized[srow..srow + nw as usize * 3]);
    }
    dst
}

/// `img_tool::resize_bilinear`, including its (src-1)/(dst-1) ratio and
/// truncating float->u8 casts.
fn resize_bilinear(src: &[u8], sw: u32, sh: u32, tw: u32, th: u32) -> Vec<u8> {
    let (sw, sh, tw, th) = (sw as usize, sh as usize, tw as usize, th as usize);
    let x_ratio = if tw > 1 { (sw - 1) as f32 / (tw - 1) as f32 } else { 0.0 };
    let y_ratio = if th > 1 { (sh - 1) as f32 / (th - 1) as f32 } else { 0.0 };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;

    let mut dst = vec![0u8; tw * th * 3];
    for y in 0..th {
        let py = y as f32 * y_ratio;
        let y0 = (py as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let yf = py - y0 as f32;
        for x in 0..tw {
            let px = x as f32 * x_ratio;
            let x0 = (px as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let xf = px - x0 as f32;
            let at = |xx: usize, yy: usize, c: usize| src[(yy * sw + xx) * 3 + c] as f32;
            for c in 0..3 {
                let top = lerp(at(x0, y0, c), at(x1, y0, c), xf);
                let bottom = lerp(at(x0, y1, c), at(x1, y1, c), xf);
                dst[(y * tw + x) * 3 + c] = lerp(top, bottom, yf) as u8;
            }
        }
    }
    dst
}

/// Decode JPEG or PNG bytes to interleaved RGB. Other containers are
/// rejected with the sniffed type in the error.
pub fn decode(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("image decode: {e}"))?
        .into_rgb8();
    let (w, h) = (img.width(), img.height());
    Ok((img.into_raw(), w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pp() -> Preprocessor {
        Preprocessor {
            align: 32,
            min_pixels: 8 * 1024,
            max_pixels: 4096 * 1024,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }

    #[test]
    fn smart_resize_matches_reference_cases() {
        let p = pp();
        // already aligned and in budget: unchanged
        assert_eq!(p.target_size(768, 768), (768, 768));
        // rounds to the nearest multiple of 32
        assert_eq!(p.target_size(770, 750), (768, 736));
        // small images scale up to the minimum pixel budget
        let (w, h) = p.target_size(20, 20);
        assert!(w * h >= p.min_pixels && w % 32 == 0 && h % 32 == 0);
        // huge images scale down under the budget, staying aligned
        let (w, h) = p.target_size(8000, 6000);
        assert!(w * h <= p.max_pixels && w % 32 == 0 && h % 32 == 0);
        // aspect ratio survives the scale-down roughly
        assert!((w as f32 / h as f32 - 8000.0 / 6000.0).abs() < 0.1);
    }

    #[test]
    fn uniform_image_survives_the_whole_pipeline() {
        let p = pp();
        // 64x64 = 4096 px sits under the 8192 min budget: the reference
        // scales it up by sqrt(2) and ceil-aligns both sides to 96
        let rgb = vec![128u8; 64 * 64 * 3];
        let (out, w, h) = p.run(&rgb, 64, 64).unwrap();
        assert_eq!((w, h), (96, 96));
        assert_eq!(out.len(), 96 * 96 * 3);
        // uniform stays uniform through bilinear upscale + normalize
        for v in &out {
            assert!((v - (128.0 / 255.0 - 0.5) / 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn padding_is_black_and_centered() {
        let p = pp();
        // 100x46 white: nearest-align gives 96x32 = 3072 px < 8192, so the
        // min-pixels branch rescales by beta=sqrt(8192/4600) -> 160x64.
        // The PAD_CEIL resize is then height-limited (scale 1.391): the
        // image becomes ceil(139.1)=140 wide, centered with 10 black
        // columns on each side.
        let rgb = vec![255u8; 100 * 46 * 3];
        let (out, w, h) = p.run(&rgb, 100, 46).unwrap();
        assert_eq!((w, h), (160, 64));
        let white = (1.0f32 - 0.5) / 0.5;
        let black = (0.0f32 - 0.5) / 0.5;
        let px = |x: u32, y: u32| out[(y * w + x) as usize]; // channel 0 plane
        assert!((px(0, 32) - black).abs() < 1e-6, "left pad column");
        assert!((px(w - 1, 32) - black).abs() < 1e-6, "right pad column");
        assert!((px(w / 2, 32) - white).abs() < 1e-6, "center is image");
        assert_eq!(out.len(), (w * h * 3) as usize);
    }
}
