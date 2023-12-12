import argparse
from src.inference import Wav2Vec2Inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="arifagustyawan/wav2vec2-large-xlsr-common_voice_13_0-id")
    parser.add_argument("--filename", type=str, default="assets/halo.wav")
    args = parser.parse_args()

    asr = Wav2Vec2Inference(args.model_name)
    transcription, confidence = asr.file_to_text(args.filename)
    print("\033[94mTranscription:\033[0m", transcription)  # Blue color for regular output
    print("\033[94mConfidence:\033[0m", confidence)
