#![no_main]
#![no_std]

use ariel_os::debug::{exit, log::info, ExitCode};
use ariel_os::time::Instant;
use microflow::model;

mod audio_sample;
mod spectrogram;
mod audio_raw;

#[cfg(all(feature = "cnn_mel", feature = "tf"))]
#[model("models/cnn_mel_tf.tflite")]
struct BirdModel;

#[cfg(all(feature = "cnn_mel", feature = "torch"))]
#[model("models/cnn_mel_torch.tflite")]
struct BirdModel;

#[cfg(all(feature = "cnn_time", feature = "tf"))]
#[model("models/cnn_time_tf.tflite")]
struct BirdModel;

#[ariel_os::task(autostart)]
async fn main() {
    info!("tiny-chirp-microflow on {}.", ariel_os::buildinfo::BOARD);
    #[cfg(feature = "cnn_mel")]
    info!("Model: cnn on mel spectrogram");
    #[cfg(feature = "cnn_time")]
    info!("Model: cnn on time domain");
    #[cfg(feature = "torch")]
    info!("Using torch produced models");
    #[cfg(feature = "tf")]
    info!("Using tensorflow produced models");
    info!(
        "Audio sample rate: {} Hz, clips={}",
        audio_sample::SAMPLE_RATE,
        audio_sample::TEST_CLIPS.len()
    );
    info!("clip | expected   | predicted  | score0 | score1");

    let mut i = 0usize;
    while i < audio_sample::TEST_CLIPS.len() {
        let start_time = Instant::now().as_micros();
        let clip = &audio_sample::TEST_CLIPS[i];

        #[cfg(feature = "cnn_mel")]
        let input = spectrogram::compute(clip.audio);
        #[cfg(feature = "cnn_time")]
        let input = audio_raw::prepare(clip.audio);

        let prediction = BirdModel::predict(input);

        let mut predicted_class: usize = 0;
        let mut best_val = prediction[(0, 0)];
        let mut c = 1usize;
        while c < prediction.ncols() {
            let v = prediction[(0, c)];
            if v > best_val {
                best_val = v;
                predicted_class = c;
            }
            c += 1;
        }

        let predicted_label = if predicted_class == 1 {
            "target----"
        } else {
            "non_target"
        };
        let end_time = Instant::now().as_micros();
        let duration = end_time - start_time;
        info!(
            "{} | {} | {} | {} | {} in {} us",
            i,
            clip.expected_label,
            predicted_label,
            prediction[(0, 0)],
            prediction[(0, 1)],
            duration
        );
        i += 1;
    }
    exit(ExitCode::SUCCESS);
}
