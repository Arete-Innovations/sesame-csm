import os
import torch
import torchaudio
import random
import pickle
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass
# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

sample_rate = 24000

prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def create_conversation():
    conversation_id = random.randint(10 ** 9, 10 ** 10 - 1)
    prompt = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        sample_rate
    )
    with open('conversations/{}.pkl'.format(conversation_id), 'wb') as f:
        pickle.dump([prompt], f)
    return conversation_id

def generate_audio(conversation_id: int, txt: str):
    with open('conversations/{}.pkl'.format(conversation_id), 'rb') as f:
        loaded_context = pickle.load(f)

    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    generator = load_csm_1b(device)
    
    # Generate requested audio
    audio_tensor = generator.generate(
        text=txt,
        speaker=0,
        context=loaded_context,
        max_audio_length_ms=10_000,
    )
    # Add spoken line to context for future use
    loaded_context.append(Segment(text=txt, speaker=0, audio=audio_tensor))
    with open('conversations/{}.pkl'.format(conversation_id), 'wb') as f:
        pickle.dump(loaded_context, f)

    return audio_tensor

def delete_conversation(conversation_id: int):
    path = f'conversations/{conversation_id}.plk'
    if os.path.exists(path):
        os.remove(path)
        return True
    else: 
        raise ValueError(f"Conversation {conversation_id} can't be found")
    return False

def main():
    conversation_id = create_conversation()
    tensor = generate_audio(conversation_id, "Hello world! I'm just happy to exist!")
    torchaudio.save(
        "single_line.wav",
        tensor.unsqueeze(0).cpu(),
        sample_rate
    )
    print("Successfully generated full_conversation.wav")

if __name__ == "__main__":
    main()
