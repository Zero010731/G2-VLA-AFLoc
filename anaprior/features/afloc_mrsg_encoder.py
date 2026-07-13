from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import torch
from torch import nn

from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch, PhraseFeatureBatch


def load_model(*args, **kwargs):
    from afloc.builder import load_model as afloc_load_model

    return afloc_load_model(*args, **kwargs)


class FrozenAFLocMRSGEncoder(nn.Module):
    def __init__(self, afloc_model: nn.Module) -> None:
        super().__init__()
        self.afloc = afloc_model.eval()
        for parameter in self.afloc.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> "FrozenAFLocMRSGEncoder":
        model = load_model(str(ckpt_path), device=device)
        return cls(model)

    def train(self, mode: bool = True) -> "FrozenAFLocMRSGEncoder":
        super().train(mode)
        self.afloc.eval()
        return self

    def encode_images(self, images: torch.Tensor) -> AFLocFeatureBatch:
        self.afloc.eval()
        with torch.no_grad():
            img_emb_l, img_emb_l2, img_emb_lf, _ = self.afloc.image_encoder_forward(images)
            image_gray = images.mean(dim=1, keepdim=True)

        return AFLocFeatureBatch(
            img_emb_l2=self._require_4d(img_emb_l2, "img_emb_l2"),
            img_emb_l=self._require_4d(img_emb_l, "img_emb_l"),
            img_emb_lf=self._require_4d(img_emb_lf, "img_emb_lf"),
            image_gray=self._require_4d(image_gray, "image_gray"),
        )

    def encode_phrases(
        self,
        phrases: Sequence[str],
        disease_descriptions: Sequence[str],
        device: Union[str, torch.device, None] = None,
    ) -> PhraseFeatureBatch:
        if len(phrases) != len(disease_descriptions):
            raise ValueError("phrases and disease_descriptions must have the same length")

        resolved_device = self._resolve_device(device)
        self.afloc.eval()
        with torch.no_grad():
            phrase_tokens = self.afloc.process_text(list(phrases), resolved_device)
            phrase_outputs = self.afloc.text_encoder_forward(
                self._require_token_batch(phrase_tokens["caption_ids"], "caption_ids"),
                self._require_token_batch(phrase_tokens["attention_mask"], "attention_mask"),
                self._require_token_batch(phrase_tokens["token_type_ids"], "token_type_ids"),
            )
            description_tokens = self.afloc.process_text(list(disease_descriptions), resolved_device)
            description_outputs = self.afloc.text_encoder_forward(
                self._require_token_batch(description_tokens["caption_ids"], "caption_ids"),
                self._require_token_batch(description_tokens["attention_mask"], "attention_mask"),
                self._require_token_batch(description_tokens["token_type_ids"], "token_type_ids"),
            )

        return PhraseFeatureBatch(
            word_embeddings=self._normalize_word_embeddings(phrase_outputs["word_embeddings"]),
            sentence_embedding=self._normalize_sentence_embedding(phrase_outputs["report_embeddings"]),
            disease_description_embedding=self._normalize_sentence_embedding(
                description_outputs["report_embeddings"]
            ),
            attention_mask=self._normalize_attention_mask(phrase_tokens["attention_mask"]),
        )

    def forward(
        self,
        images: torch.Tensor,
        phrases: Sequence[str],
        disease_descriptions: Sequence[str],
        device: Union[str, torch.device, None] = None,
    ) -> tuple[AFLocFeatureBatch, PhraseFeatureBatch]:
        phrase_device = device if device is not None else images.device
        return (
            self.encode_images(images),
            self.encode_phrases(phrases, disease_descriptions, device=phrase_device),
        )

    def _resolve_device(self, device: Union[str, torch.device, None]) -> Union[str, torch.device]:
        if device is not None:
            return device

        parameter = next(self.afloc.parameters(), None)
        if parameter is not None:
            return parameter.device
        return torch.device("cpu")

    @staticmethod
    def _require_4d(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B,C,H,W]")
        return tensor.detach()

    @staticmethod
    def _require_token_batch(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [B,T]")
        return tensor

    @staticmethod
    def _normalize_word_embeddings(word_embeddings: torch.Tensor) -> torch.Tensor:
        if word_embeddings.ndim != 3:
            raise ValueError("word_embeddings must have shape [B,D,T]")
        return word_embeddings.detach().transpose(1, 2).contiguous()

    @staticmethod
    def _normalize_sentence_embedding(sentence_embedding: torch.Tensor) -> torch.Tensor:
        if sentence_embedding.ndim == 1:
            sentence_embedding = sentence_embedding.unsqueeze(0)
        if sentence_embedding.ndim != 2:
            raise ValueError("sentence_embedding must have shape [B,D]")
        return sentence_embedding.detach()

    @staticmethod
    def _normalize_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B,T]")
        return attention_mask.detach().to(dtype=torch.bool)
