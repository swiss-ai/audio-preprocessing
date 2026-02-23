import torchaudio
import sys
import os

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

sys.modules["transformer_engine"] = None
sys.modules["modelopt"] = None
sys.modules["torchao"] = None

from typing import Union,Optional, Dict, Any, Tuple, List

import torch
import soundfile as sf
import numpy as np
import json

import nemo.collections.asr as nemo_asr

from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from faster_whisper import WhisperModel


class AudioProcessing:
    def __init__(self, lid: Optional[str]=None):
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache directory
        cache_dir = "/capstor/store/cscs/swissai/infra01/MLLM/audioprocessing"
        os.makedirs(cache_dir, exist_ok=True) 
        torch.hub.set_dir(cache_dir)

        # sample rate
        self.sr = 16000  # Required value for VAD

        # Load VAD model
        self.vad_model = self.load_vad_model()

        # Load speaker diarization model
        self.speaker_encoder = self.load_speaker_model()

        # Load LID
        # lid is None if language not known as language is known 
        self.lid = lid
        self.lid_model = self.load_lid_model(lid)
        
        # Transcription Model
        self.transcription_model = self.load_transcription_model()

        # Maps Language code of LID to transcription model
        self.mapper = self.load_mapper()

        # Annotation Model
        self.annotation_model = self.load_annotation_model(cache_dir = cache_dir)

    def load_vad_model(self, **kwargs):
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=False,
                                onnx=True)
        get_timestamps = utils[0]

        return get_timestamps, vad_model

    def load_annotation_model(self, **kwargs):
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from qwen_omni_utils import process_mm_info
        
        path = kwargs.get("cache_dir")

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                path,
                dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        processor = Qwen3OmniMoeProcessor.from_pretrained(path)

        return model, processor
    
    def load_speaker_model(self, **kwargs):
        speaker_encoder = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name='titanet_large'
            ).to(self.device)
        speaker_encoder.eval()

        return speaker_encoder

    def load_lid_model(self, lid: Optional[str] = None):
        if(lid is None):
            lid_model_name = "facebook/mms-lid-126"
            lid_processor = AutoFeatureExtractor.from_pretrained(lid_model_name)
            lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(lid_model_name).to(self.device)
            lid_model.eval()
        else:
            lid_processor = None
            lid_model = None

        return lid_processor, lid_model

    def load_transcription_model(self):
        return WhisperModel("large-v3", device="cuda", compute_type="float16")

    def load_mapper(self) -> dict:
        import json
        with open("mms_to_whisper.json", "r") as f:
            return json.load(f)

    def load_audio(self, 
                   audio: Union[str,np.ndarray, torch.Tensor],
                   sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        
        if isinstance(audio, str):
            wav, sr = sf.read(audio)
            wav = torch.from_numpy(wav).float()
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio).float()
        elif isinstance(audio, torch.Tensor):
            wav = audio.float()
        
        if sr is None:
            raise ValueError("Sample rate not given.")

        if wav.ndim > 1:
            wav = wav.mean(dim=1)
        
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sr)

        return wav.squeeze(0), sr, self.sr
    
    def get_embeddings(self, wav: torch.Tensor, segments: list) -> np.ndarray:
        embeddings = []

        for seg in segments:
            start = int(seg['start'])
            end = int(seg['end'])
            seg_audio = wav[start:end]

            # (B, T)
            seg_audio = seg_audio.unsqueeze(0).to(self.device)
            length = torch.tensor([seg_audio.shape[1]]).to(self.device)

            with torch.no_grad():
                emb, _ = self.speaker_encoder.forward(input_signal=seg_audio, input_signal_length=length)
                embeddings.append(emb.squeeze().cpu().numpy())
            
        return np.array(embeddings)
    
    def detect_language(self, speaker_audio: torch.Tensor) -> Tuple[str, float]:
        speaker_audio = speaker_audio.cpu().numpy()
        
        lid_processor, lid_model = self.lid_model

        inputs = lid_processor(speaker_audio, sampling_rate=self.sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = lid_model(**inputs).logits

        probs = torch.softmax(outputs, dim=-1)

        confidence, lang_id = torch.max(probs, dim=-1)
        lang = lid_model.config.id2label[lang_id.item()]
        return lang, confidence.item()
    
    def estimate_speakers(self, embeddings: torch.Tensor, 
                          max_speakers: Optional[int] = 10) -> int:

        affinity = cosine_similarity(embeddings) # Affinity Matrix
        affinity = (affinity + 1) / 2  # [-1,1] -> [0,1]
        
        laplacian = np.diag(np.sum(affinity, axis=1)) - affinity

        vals, _ = np.linalg.eigh(laplacian)
        vals = np.sort(vals)

        max_gap = 0
        est_speakers = 1
        n_neighbours = max(1,min(len(vals)-1, max_speakers))
        for k in range(1, n_neighbours):
            gap = vals[k+1] - vals[k]
            if gap > max_gap:
                max_gap = gap
                est_speakers = k + 1 
        return est_speakers, n_neighbours

    def transcribe(self, wav: np.ndarray, vad_segments: list) -> list:
        for seg in vad_segments:
            start_sample = int(seg['start'])
            end_sample = int(seg['end'])
            
            audio_slice = wav[start_sample:end_sample]
            audio_slice = audio_slice.cpu().numpy()
            audio_slice = np.squeeze(audio_slice)
            audio_slice = audio_slice.astype(np.float32)

            if len(audio_slice) == 0:
                seg['text'] = ""
                continue
            segments_generator, info = self.transcription_model.transcribe(
                audio_slice, 
                language= self.mapper[seg['language']],
                condition_on_previous_text=False  
            )
            
            text_chunks = []
            for w_seg in segments_generator:
                text_chunks.append(w_seg.text)
                
            seg['text'] = " ".join(text_chunks).strip()
            
        return vad_segments

    def vad(self, wav):
        get_timestamps, vad_model = self.vad_model
        vad_segments = get_timestamps(wav, vad_model, sampling_rate=self.sr)
        return vad_segments

    def initialise_metadata(self, vad_segments):
        for seg in vad_segments:
            seg['speaker'] = 0
            seg['language'] = self.lid if self.lid else "unknown"
            seg['score'] = 1.0 if self.lid else 0.0
        
        return ("language", "speaker")

    def update_metadata(self, wav, vad_segments, **kwargs):

        # Arguments
        valid_vad_segment_min_dur = kwargs.get("valid_vad_segment_min_dur", 1)
        num_speakers = kwargs.get("num_speakers", None)
        similarity_threshold = kwargs.get("similarity_threshold", 0.72)
        lid_merge_duration = kwargs.get("lid_merge_duration", 30)


        # Get embeddings for each segment
        embeddings = self.get_embeddings(wav, vad_segments)
        if len(embeddings) == 0: return

        # Separate long and short segments
        min_samples = valid_vad_segment_min_dur  * self.sr
        long_idx = []
        short_idx = []

        for i, seg in enumerate(vad_segments):
            if (seg['end'] - seg['start']) >= min_samples: long_idx.append(i)
            else: short_idx.append(i)
        
        if len(long_idx) == 0:
            long_idx = short_idx
            short_idx = []
        
        long_embeddings = embeddings[long_idx]
        
        #------------Estimate Number of speakers-----------------
        # Use only long embeddings for selecting the speakers to 
        # prevent overestimation

        # Number of speakers is unknown
        if num_speakers is None: 
            n_clusters, n_neighbors = self.estimate_speakers(long_embeddings) 
        # Number of speakers is known
        else:
            n_clusters = num_speakers

        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=max(1, min(10, len(long_embeddings) - 1)),
            assign_labels='kmeans',
            random_state=42
        )
        labels = clusterer.fit_predict(long_embeddings)

        #------------Update Speakers-----------------

        # Long segments
        for i, label in zip(long_idx, labels):
            vad_segments[i]['speaker'] = label

        # Short segments
        if len(short_idx) > 0:
            unique_speakers = np.unique(labels)
            centroids = []
            
            # Calculate the average embedding for each speaker
            for spk in unique_speakers:
                spk_embs = long_embeddings[labels == spk]
                centroid = np.mean(spk_embs, axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            short_embeddings = embeddings[short_idx]

            sim_matrix = cosine_similarity(short_embeddings, centroids)
            best_matching_speaker_indices = np.argmax(sim_matrix, axis=1)
            
            for i, idx in enumerate(short_idx):
                vad_segments[idx]['speaker'] = unique_speakers[best_matching_speaker_indices[i]]

        #------------Update Language-----------------
        # Stitch segments of same speaker until length 
        # is long enough for language identification

        if(self.lid is None):
            speaker_langs = {}
            speakers = np.unique(labels)

            for spkr in speakers:
                cur_spkr_seg = []
                cur_spkr_len = 0
                for seg in vad_segments:
                    if(seg['speaker'] == spkr):
                        seg_wav = wav[seg['start']:seg['end']]
                        cur_spkr_seg.append(seg_wav)
                        cur_spkr_len += seg_wav.shape[0]
                    
                    if(cur_spkr_len >= lid_merge_duration * self.sr):
                        break

                if cur_spkr_seg:
                    combined = torch.cat(cur_spkr_seg)
                    lang, score = self.detect_language(combined)
                    speaker_langs[spkr] = (lang,score)
                else:
                    speaker_langs[spkr] = ("unknown",0.0)
            
            for seg in vad_segments:
                lang, score = speaker_langs.get(seg['speaker'], ("unknown", 0.0))
                seg['language'] = lang
                seg['score'] = score

    def process_speech_attributes(self, wav, vad_segments, **kwargs):
        parameter = kwargs.get("parameter", None)
        merge_threshold_duration = kwargs.get("merge_threshold_duration", 1.0)
        sr = kwargs.get("sr", self.sr)
        max_duration = kwargs.get("max_duration", 10.0)

        p1, p2 = parameter
        merged_segments = []
        
        cur_seg = vad_segments[0]
        for next_seg in vad_segments[1:]:
            gap = next_seg['start'] - cur_seg['end']
            
            same_lang = cur_seg[p1] == next_seg[p1]
            same_speaker = cur_seg[p2] == next_seg[p2]

            duration = next_seg['end'] - cur_seg['start']

            if(gap <=  (merge_threshold_duration * self.sr) and 
                same_lang and
                same_speaker and 
                duration <= max_duration * self.sr):
               cur_seg['end'] = next_seg['end']
            else:
                merged_segments.append(cur_seg)
                cur_seg = next_seg
        
        merged_segments.append(cur_seg)

        merged_segments = self.transcribe(wav, merged_segments)

        speech_attribute = []
        rescale_time = 1#sr/self.sr

        for seg in merged_segments:
            speech_attribute.append({
                "start": seg['start'] / self.sr * rescale_time,
                "end": seg['end'] / self.sr * rescale_time,
                "duration": (seg['end'] - seg['start'])/ self.sr * rescale_time,
                "speaker": f"Speaker {seg['speaker']}",
                "language": seg['language'],
                "score": seg['score'],
                "text": seg["text"]
            })
        
        return speech_attribute
    
    def process_non_speech_attributes(self, wav, **kwargs):
        model, processor = self.annotation_model
        
        non_speech_prompt = """
            Analyze the provided audio and extract detailed non-speech annotations. 
            Focus on the overall acoustic environment, any discrete sound events, and musical characteristics.

            Output the results strictly as a valid JSON object using the exact schema below. 
            Do not include any markdown formatting (such as ```json), introductory text, or explanations. 

            {
                "acoustic_environment": "string or null (Describe the overall background setting, e.g., 'busy street', 'nature', 'quiet room')",
                "audio_events": ["list of strings (List specific isolated sounds heard, e.g., 'door slam', 'dog barking', 'siren')"],
                "music_metadata": {
                "is_music": "boolean (true if any music is present, false otherwise)",
                "genre": "string or null (e.g., 'jazz', 'electronic', 'orchestral', or null if no music is present)",
                "instruments": ["list of strings (e.g., 'piano', 'electric guitar', 'drums', or empty list if no instruments are detected)"]
                }
            }

            """

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "in_memory_tensor"}, 
                    {"type": "text", "text": non_speech_prompt}
                ],
            },
        ]

        text_input = processor.apply_chat_template(conversation, 
                                                add_generation_prompt=True, 
                                                tokenize=False)
        
        audios = [wav.squeeze().cpu().numpy()]

        inputs = processor(text=text_input, 
                            audio=audios,
                            return_tensors="pt", 
                            padding=True, 
                            use_audio_in_video=False).to(model.device).to(model.dtype) 
        
        text_ids, audio = model.generate(**inputs, 
                                 thinker_return_dict_in_generate=True)

        generated_text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)[0]

        clean_text = generated_text.strip("`").removeprefix("json").strip()
        annotations = json.loads(clean_text)
        
        return annotations

    def merge(self, wav, vad_segments, **kwargs):
        speech_attribute = None
        non_speech_attribute = None


        if(self.audio_type == "speech"):
            speech_attribute = self.process_speech_attributes(wav, 
                                                        vad_segments, 
                                                        **kwargs)
        else:
            non_speech_attribute = self.process_non_speech_attributes(wav, 
                                                             **kwargs
                                                            )

        final_output = [{
            "audio_type": self.audio_type,
            "duration": wav.numel()/ self.sr,
            "sample_rate": self.sr,
            "speech_attributes": speech_attribute,
            "non_speech_attributes": non_speech_attribute
        }]
        
        return final_output
    
    def classify_audio(self, wav, vad_segments, ratio_thresh=0.05):
        duration = wav.numel()
        speech_duration = sum(seg['end'] - seg['start'] for seg in vad_segments)

        ratio = speech_duration/duration
        if(ratio >= ratio_thresh):
            return "speech"
        else:
            return "non_speech"

    def process(self, 
                audio: Union[str, np.ndarray, torch.Tensor],
                sr: Optional[int] = None, 
                num_speakers: Optional[int] = None,
                similarity_threshold: Optional[float] = 0.72,
                lid_merge_duration: Optional[float] = 30, #in seconds
                merge_threshold_duration: Optional[float] = 1.0, #in seconds
                max_duration: Optional[float] = 10.0, #in seconds
                valid_vad_segment_min_dur: Optional[float] = 1, #in seconds
                **kwargs) -> List[Dict[str,Any]]:
        
        # Load Audio
        wav, ip_sr, op_sr = self.load_audio(audio, sr)

        # Perform VAD
        vad_segments = self.vad(wav)

        # Classify Speech and Non-speech
        if not vad_segments:
            self.audio_type = "non_speech"
        else:
            self.audio_type = self.classify_audio(wav, vad_segments, 0.05)
        
        parameters = None
        if(self.audio_type == "speech"):
            # Initialise additional metadata (Language, Speaker, ...)
            parameters = self.initialise_metadata(vad_segments)

            # Update Metadata (LID + Speaker Identification)
            self.update_metadata(wav, vad_segments, **kwargs)

        # Merge segments of conditioned on parameters/metadata 
        # if separation less than merge_threshold_duration
        # and total length less than max_duration
        output = self.merge(wav,
                            vad_segments, 
                            parameter = parameters,
                            merge_threshold_duration = merge_threshold_duration,
                            sr = ip_sr)

        return output


if __name__ == "__main__":
    import time
    t0 = time.time()
    ap = AudioProcessing()
    print("Intialiase Object : ", time.time() - t0)
    path = "multiplespeaker.mp3"
    # path = "sample_0004_original.wav"
    # path = "sample-merged.wav"
    # path = "mergedaud.wav"
    # path = "sample_0002_original.wav"

    t0 = time.time()
    op = ap.process(path)
    print("Process : ", time.time() - t0)

    import json
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(op, f, indent=4, ensure_ascii=False)
