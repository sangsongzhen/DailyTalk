import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PostNet, VarianceAdaptor, ConversationalContextEncoder, RoleStyleEncoder
from utils.tools import get_mask_from_lengths
from .transformers.blocks import FiLM


class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config

        if model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "lstransformer":
        #     from .transformers.lstransformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "fastformer":
        #     from .transformers.fastformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "conformer":
        #     from .transformers.conformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "reformer":
        #     from .transformers.reformer import TextEncoder, Decoder
        else:
            raise ValueError("Unsupported Block Type: {}".format(model_config["block_type"]))

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.mix_proj = nn.Linear(512, model_config["transformer"]["encoder_hidden"])  # 512 → 256
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = self.emotion_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],
            )
        
        # 语音角色风格编码器
        self.use_role_encoder = model_config["style_encoder"]["use_role_encoder"]
        self.use_role_encoder_syn = model_config["style_encoder"]["use_role_encoder_synthesize"]

        if self.use_role_encoder:
            self.role_style_encoder = RoleStyleEncoder(model_config)
        
        self.history_type = model_config["history_encoder"]["type"]

        if self.history_type != "none":
            if self.history_type == "Guo":
                self.context_encoder = ConversationalContextEncoder(preprocess_config, model_config)
        
        # FiLM
        self.use_film = model_config["film"]["use_film"]
        self.film = FiLM()
        self.film_mlp = nn.Sequential(
            nn.Linear(model_config["transformer"]["encoder_hidden"], model_config["transformer"]["encoder_hidden"] * 2),
            nn.ReLU(),
            nn.Linear(model_config["transformer"]["encoder_hidden"] * 2, model_config["transformer"]["encoder_hidden"] * 2),
        )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        emotions=None,
        history_info=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        # texts, text_embeds = self.encoder(texts, src_masks)

        # Context Encoding
        context_encodings = None
        role_style_vec =None
        if self.history_type != "none":
            if self.history_type == "Guo":
                (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                    history_audio_embs,
                    history_audio_lens
                ) = history_info

                if self.use_role_encoder_syn:
                    role_style_vec = self.role_style_encoder(history_audio_embs, history_audio_lens)
                context_encodings = self.context_encoder(
                    text_embs,
                    speakers,
                    history_text_embs,
                    history_speakers,
                    history_lens,
                )

        # 拼接文本上下文向量和角色风格向量
        mix_encodings = None
        if context_encodings is not None and role_style_vec is not None:
            mix_encodings = torch.cat([context_encodings, role_style_vec], dim=-1)  # [B, 512]
        elif context_encodings is not None:
            mix_encodings = context_encodings  # [B, 256]
        elif role_style_vec is not None:
            mix_encodings = role_style_vec     # [B, 256]

        # 如果拼接后是512维，做降维
        if mix_encodings is not None and mix_encodings.shape[-1] == 512:
            mix_encodings = self.mix_proj(mix_encodings)
        
        if self.use_film:
            if mix_encodings is not None:
            # 假设 mix_encodings: [B, H]
                gamma_beta = self.film_mlp(mix_encodings)       # [B, 2H]
                gammas, betas = gamma_beta.chunk(2, dim=-1)     # [B, H], [B, H]
                gammas = gammas.unsqueeze(1)  # → [B, 1, H]
                betas = betas.unsqueeze(1)    # → [B, 1, H]
                mix_encodings = mix_encodings.unsqueeze(1)      # [B, 1, H]

                mix_encodings = self.film(mix_encodings, gammas, betas)  # [B, 1, H]
                mix_encodings = mix_encodings.squeeze(1)                 # → [B, H]
        else:
            gammas = betas = None


        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds) # [B, H]

        emotion_embeds = None
        if self.emotion_emb is not None:
            emotion_embeds = self.emotion_emb(emotions)

        # FiLM Encoder
        texts, text_embeds = self.encoder(texts, src_masks, gammas, betas, use_film = self.use_film)

        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
        ) = self.variance_adaptor(
            speaker_embeds,
            emotion_embeds,
            # context_encodings,
            mix_encodings,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            p_control,
            e_control,
            d_control,
            step,
        )

        # FiLM Decoder
        output, mel_masks = self.decoder(output, mel_masks, gammas, betas, use_film = self.use_film)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            p_targets,
            e_targets,
        )
