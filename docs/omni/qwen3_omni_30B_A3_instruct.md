### Ohayou

1. model.thinker

<!-- 
model.thinker.audio_tower # audio encoder

model.thinker.visual # visual encoder

model.thinker.model # decoder with self_attn + experts
 -->


2. model.talker
<!-- # model.talker
model.talker.text_projection # MLP for text data
model.talker.hidden_projection # MLP for MM data
# operation combine(hidden_projection, text_projection)  
model.talker.model # main decoding block with self_attn and experts inside based on the combined (MM_MLP, text_MLP)
model.talker.codec_head # receives the last_hidden_state from talker.model and produce only 1 logit (special acoustic token) out [1, 1, 1024] (batch_size, token_num, hidden_size)
model.talker.code_predictor # takes as input from [talker.codec_head(1_acoustic_token) + talker.model.last_hidden_state] and generate 15 aco_tokens + first_head_token   -> that will be passed to code2wav model
# code predtictor takes 2 inputs
#    # last_hidden_state from talker.model [1, 1, 1024]
#    # special_acoustic_token from codec_head [1, 1, 1024]
#    combines them cat([1, 1, 1024], [1, 1, 1024]) -> [1, 2, 1024]
#    produces are 15 acoustic tokens -->


3. model.code2wav

<!-- # from code_predictor it receives the [1, 16, 184] [batch, aco_tokens, talker.seq_len]
# The Idea is: we take (discrete tokens representing sound) and convert into a continuous audio waveform -->






#### Draft

inference example

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
import soundfile as sf

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "/home/vladimir_albrekht/projects/2025_sep_22_qwen3omni/models/Qwen3-Omni-30B-A3B-Instruct"
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/home/vladimir_albrekht/projects/2025_sep_22_qwen3omni/models/Qwen3-Omni-30B-A3B-Instruct/unity_codes/audio_chunks/chunk_1.wav"},
            {"type": "text", "text": "Transcribe the English audio into text."}
        ],
    },
]

# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = False

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, 
                                 speaker="Chelsie", 
                                 thinker_return_dict_in_generate=True,
                                 use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
print(text)
if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
```


##### Some new interesting pytorch tricks:

1. Bolean mask tensor

```python
def _get_talker_user_parts(
        self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
    ):
        user_talker_part = torch.empty(
            (1, segment_end_index - im_start_index, self.config.talker_config.text_config.hidden_size),
            device=self.talker.device,
            dtype=self.talker.dtype,
        )

        user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]

        # Multimodal data exists
        if user_mm_mask.any():
            user_thinker_hidden_mm = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
            mm_hidden = self.talker.hidden_projection(user_thinker_hidden_mm).to(self.talker.device)
            user_talker_part[user_mm_mask] = mm_hidden
        user_thinker_embed = thinker_embed[:, im_start_index:segment_end_index][~user_mm_mask]
        user_text_hidden = self.talker.text_projection(user_thinker_embed).to(self.talker.device)
        user_talker_part[~user_mm_mask] = user_text_hidden
        return user_talker_part
```

