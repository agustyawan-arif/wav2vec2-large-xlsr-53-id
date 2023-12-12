# Wav2Vec2 Large XLSR Common Voice Indonesia

This repository contains code for performing audio transcription using the Wav2Vec2 model trained on the Common Voice dataset 13 for Indonesian.
[![Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/arifagustyawan/wav2vec2-large-xlsr-common_voice_13_0-id)
[![Model Card](https://img.shields.io/badge/Hugging%20Face-Model%20Card-blueviolet)](https://huggingface.co/arifagustyawan/wav2vec2-large-xlsr-common_voice_13_0-id)

https://github.com/agustyawan-arif/wav2vec2-large-xlsr-common_voice_13_0-id/assets/82918938/bddc6840-e824-4066-957d-4b04e5696940

(_🔈 sound on_ gradio speech to text simulation)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/agustyawan-arif/wav2vec2-large-xlsr-common_voice_13_0-id.git
   cd wav2vec2-large-xlsr-common_voice_13_0-id
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Pre-trained model:
   It will automatically download the model from huggingface model hub

## Usage

### Inference

To transcribe an audio file using the pre-trained model, use the following command:

```bash
python wrapper.py --model_name arifagustyawan/wav2vec2-large-xlsr-common_voice_13_0-id --filename path/to/your/audio/file.wav
```

Replace `path/to/your/audio/file.wav` with the path to the audio file you want to transcribe. Sample are available in `assets` directory

The transcribed text and confidence score will be printed to the console.

## Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 5.0656        | 2.88  | 400  | 2.7637          | 1.0    |
| 1.1404        | 5.76  | 800  | 0.4483          | 0.6088 |
| 0.3698        | 8.63  | 1200 | 0.4029          | 0.5278 |
| 0.2695        | 11.51 | 1600 | 0.3976          | 0.5036 |
| 0.2074        | 14.39 | 2000 | 0.3988          | 0.4793 |
| 0.1796        | 17.27 | 2400 | 0.3952          | 0.4590 |
| 0.1523        | 20.14 | 2800 | 0.3986          | 0.4463 |
| 0.1352        | 23.02 | 3200 | 0.4143          | 0.4374 |
| 0.121         | 25.9  | 3600 | 0.4022          | 0.4337 |
| 0.1085        | 28.78 | 4000 | 0.4115          | 0.4316 |

## Future Updates

- [ ] Improve the training script: If there are new techniques or improvements for training the Wav2Vec2 model, consider updating the training script.

- [ ] Update dependencies: Regularly check for updates to the libraries used in the project and update the `requirements.txt` file.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Common voice Datasets](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)

Feel free to contribute to this project by opening issues or pull requests.

Happy coding!
