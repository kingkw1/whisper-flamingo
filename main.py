import subprocess

model_checkpoint = "models/whisper-flamingo_en-x_small.pt"
avhubert_ckpt = "models/large_noise_pt_noise_ft_433h_only_weights.pt"
avhubert_path = "av_hubert/avhubert/"
noise_fn_path = "test_data/my_test.tsv"
output_transcription_dir = "decode/mytest/"

cmd = [
    "python", "-u", "whisper_decode_video.py",
    "--lang", "en",
    "--model-type", "small",
    "--modalities", "avsr",
    "--checkpoint-path", model_checkpoint,
    "--av-hubert-path", avhubert_path,
    "--av-hubert-ckpt", avhubert_ckpt,
    "--decode-path", output_transcription_dir,
    "--fp16", "0",
    "--use_av_hubert_encoder", "1",
    "--av_fusion", "separate",
    "--user-tsv", noise_fn_path,
    # "--modalities", "asr",
    # "--use_av_hubert_encoder", "0",
    # "--av_fusion", "None",
]


subprocess.run(cmd)
print(f"Transcription saved under directory {output_transcription_dir}")
