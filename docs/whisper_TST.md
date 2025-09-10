# Whisper Target Speaker Transcription


https://chatgpt.com/c/68c112be-9250-832c-bd09-a855eddba42f

Whisper that can Transcribe only target speaker when there is noisy or other speakers on the background


Input:
Audio (with different speakers)

Output: 
Transcription with target speaker.



Topics:

- Target Speech Extraction (TSE)
- Target-Speaker Voice Activity Detection (TS-VAD)
- Target-Speaker ASR (TS-ASR)
- Personalized ASR


If before ASR:

Target Speech Extraction (TSE) / VoiceFilter / SpeakerBeam: separate out one desired speaker given a short “enrollment” clip of them. Google’s VoiceFilter-Lite is a classic on-device approach; NTT’s SpeakerBeam is a strong research line. Use this before ASR.