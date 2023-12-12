import os
import sys
import torch
import argparse
from halo import Halo
import soundfile as sf
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

path_this_file = os.path.dirname(os.path.abspath(__file__))
pat_project_root = os.path.join(path_this_file, "..")
sys.path.append(pat_project_root)

class Wav2Vec2Inference:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        """
        Initializes the class with the provided parameters.

        Args:
            model_name (str): The name of the model to be used.
            hotwords (list, optional): A list of hotwords. Defaults to an empty list.
            use_lm_if_possible (bool, optional): Specifies whether to use a language model if possible. 
                Defaults to True.
            use_gpu (bool, optional): Specifies whether to use the GPU. Defaults to True.

        Returns:
            None
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if use_lm_if_possible:            
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible

    def buffer_to_text(self, audio_buffer):
        """
        Transcribes the given audio buffer into text.

        Args:
            audio_buffer (list): A list representing the audio buffer.

        Returns:
            tuple: A tuple containing the transcribed text (str) and the confidence score (float).
        """
        spinner = Halo(text="Transcribing audio...", spinner="dots")
        spinner.start()

        try:
            if len(audio_buffer) == 0:
                return ""

            inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device),
                                    attention_mask=inputs.attention_mask.to(self.device)).logits            

            if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
                transcription = \
                    self.processor.decode(logits[0].cpu().numpy(),                                      
                                          hotwords=self.hotwords, 
                                          output_word_offsets=True,                                      
                                       )                             
                confidence = transcription.lm_score / len(transcription.text.split(" "))
                transcription = transcription.text       
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                confidence = self.confidence_score(logits,predicted_ids)

            spinner.succeed("Audio transcribed successfully!")
            return transcription, confidence.item()
        except Exception as e:
            spinner.fail(f"Error during transcription: {str(e)}")
            return "", 0.0

    def confidence_score(self, logits, predicted_ids):
        """
        Calculate the confidence score for the predicted IDs based on the logits.

        Parameters:
            logits (torch.Tensor): The logits tensor.
            predicted_ids (torch.Tensor): The predicted IDs tensor.

        Returns:
            float: The average confidence score for the predicted IDs.
        """
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self, filename):
        """
        Reads an audio file and converts it to text using the buffer_to_text method.

        Parameters:
            filename (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the transcription (str) and the confidence (float) of the transcription. If there is an error reading the audio file, an empty string and a confidence of 0.0 will be returned.
        """
        spinner = Halo(text="Reading audio file...", spinner="dots")
        spinner.start()

        try:
            audio_input, samplerate = sf.read(filename)
            assert samplerate == 16000
            transcription, confidence = self.buffer_to_text(audio_input)
            spinner.succeed("File read successfully!")
            return transcription, confidence
        except Exception as e:
            spinner.fail(f"Error reading audio file: {str(e)}")
            return "", 0.0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="arifagustyawan/wav2vec2-large-xlsr-common_voice_13_0-id")
    parser.add_argument("--filename", type=str, default="assets/halo.wav")
    args = parser.parse_args()

    with Halo(text="Initializing Wav2Vec2 Inference...", spinner="dots") as init_spinner:
        try:
            asr = Wav2Vec2Inference(args.model_name)
            init_spinner.succeed("Wav2Vec2 Inference initialized successfully!")
        except Exception as e:
            init_spinner.fail(f"Error initializing Wav2Vec2 Inference: {str(e)}")
            sys.exit(1)

    with Halo(text="Performing audio transcription...", spinner="dots") as transcribe_spinner:
        transcription, confidence = asr.file_to_text(args.filename)
        
    print("\033[94mTranscription:\033[0m", transcription)
    print("\033[94mConfidence:\033[0m", confidence)
    