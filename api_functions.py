import os
import torch
import torchaudio
import random
import pickle
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass
from run_csm.py import prepare_prompt
# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

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

def create_conversation():
    conversation_id = random.randint(10 ** 9, 10 ** 10 - 1)
    prompt = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.sample_rate
    )
    with open('conversations/{}.pkl'.format(conversation_id), 'wb') as f:
        pickle.dump([prompt], f)
    return conversation_id
