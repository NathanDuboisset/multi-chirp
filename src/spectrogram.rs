use libm::{cosf, log10f, logf, powf, sqrtf};
use microflow::buffer::Buffer4D;
use nalgebra::SMatrix;

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_LENGTH: usize = 1024;
pub const FRAME_STEP: usize = 256;
pub const FFT_BINS: usize = 512;
pub const MEL_BINS: usize = 80;
pub const FRAMES: usize = 184;
const EPS: f32 = 1e-6;

type SpectrogramImage = SMatrix<[f32; 1], FRAMES, MEL_BINS>;

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * log10f(1.0 + hz / 700.0)
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (powf(10.0, mel / 2595.0) - 1.0)
}

fn hann_window(i: usize) -> f32 {
    let phase = (2.0 * core::f32::consts::PI * i as f32) / (FRAME_LENGTH as f32 - 1.0);
    0.5 - 0.5 * cosf(phase)
}

fn create_mel_bin_edges() -> [usize; MEL_BINS + 2] {
    let mut bins = [0usize; MEL_BINS + 2];
    let low_mel = hz_to_mel(80.0);
    let high_mel = hz_to_mel(8000.0);

    let mut i = 0;
    while i < MEL_BINS + 2 {
        let frac = i as f32 / (MEL_BINS + 1) as f32;
        let mel = low_mel + frac * (high_mel - low_mel);
        let hz = mel_to_hz(mel);
        let bin = ((FRAME_LENGTH as f32 + 1.0) * hz / SAMPLE_RATE as f32) as usize;
        bins[i] = if bin >= FFT_BINS { FFT_BINS - 1 } else { bin };
        i += 1;
    }
    bins
}

pub fn compute(audio: &[f32]) -> Buffer4D<f32, 1, FRAMES, MEL_BINS, 1> {
    let mel_edges = create_mel_bin_edges();
    let mut out: SpectrogramImage = SMatrix::from_element([0.0f32; 1]);

    let mut frame_idx = 0usize;
    while frame_idx < FRAMES {
        let mut frame = [0f32; FRAME_LENGTH];
        let base = frame_idx * FRAME_STEP;

        let mut n = 0usize;
        while n < FRAME_LENGTH {
            let sample = if base + n < audio.len() { audio[base + n] } else { 0.0 };
            frame[n] = sample * hann_window(n);
            n += 1;
        }

        let spectrum = microfft::real::rfft_1024(&mut frame);
        let mut magnitudes = [0f32; FFT_BINS];
        let mut b = 0usize;
        while b < FFT_BINS {
            let c = spectrum[b];
            magnitudes[b] = sqrtf(c.re * c.re + c.im * c.im);
            b += 1;
        }

        let mut m = 0usize;
        while m < MEL_BINS {
            let left = mel_edges[m];
            let center = mel_edges[m + 1];
            let right = mel_edges[m + 2];
            let mut mel_energy = 0.0f32;

            let mut k = left;
            while k < center {
                let w = (k - left) as f32 / (center - left).max(1) as f32;
                mel_energy += magnitudes[k] * w;
                k += 1;
            }

            k = center;
            while k < right {
                let w = (right - k) as f32 / (right - center).max(1) as f32;
                mel_energy += magnitudes[k] * w;
                k += 1;
            }

            let log_mel = logf(mel_energy + EPS);
            out[(frame_idx, m)] = [log_mel];
            m += 1;
        }

        frame_idx += 1;
    }

    [out]
}
