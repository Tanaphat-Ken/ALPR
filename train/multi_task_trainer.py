"""
Multi-task trainer for TrOCR with province classification.

This trainer handles both text recognition and province classification tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput
from multi_task_trocr import MultiTaskTrOCR, province_code_to_id, id_to_province_code

logger = logging.getLogger(__name__)


class MultiTaskTrOCRTrainer(Seq2SeqTrainer):
    """
    Trainer for multi-task TrOCR that handles both text and province prediction.
    """
    
    def __init__(self, model: MultiTaskTrOCR, **kwargs):
        super().__init__(model=model, **kwargs)
        self.model = model
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss for both text and province prediction tasks.
        """
        # Extract province labels if available
        province_labels = inputs.pop("province_labels", None)
        
        # Forward pass
        outputs = model(**inputs, province_labels=province_labels)
        
        # The model returns combined loss
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step for both text and province prediction.
        """
        has_labels = "labels" in inputs
        has_province_labels = "province_labels" in inputs
        
        inputs = self._prepare_inputs(inputs)
        
        # Extract relevant inputs
        pixel_values = inputs.get("pixel_values")
        labels = inputs.get("labels") if has_labels else None
        province_labels = inputs.get("province_labels") if has_province_labels else None
        
        with torch.no_grad():
            if self.args.predict_with_generate and not prediction_loss_only:
                # Use generate method for text and province prediction
                generation_outputs = model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=20,
                    num_beams=1,
                    do_sample=False,
                )
                
                generated_tokens = generation_outputs['sequences']
                province_predictions = generation_outputs['province_predictions']
                
                # Pad generated tokens to match labels length if needed
                if labels is not None:
                    # Get pad token id safely
                    pad_token_id = getattr(self.tokenizer, 'pad_token_id', None) or 0
                    
                    max_length = max(generated_tokens.size(-1), labels.size(-1))
                    if generated_tokens.size(-1) < max_length:
                        padding = torch.full(
                            (generated_tokens.size(0), max_length - generated_tokens.size(-1)),
                            pad_token_id,
                            dtype=generated_tokens.dtype,
                            device=generated_tokens.device,
                        )
                        generated_tokens = torch.cat([generated_tokens, padding], dim=-1)
                
                # Combine text and province predictions
                # For text: use generated tokens
                # For province: add province predictions as additional "tokens"
                if province_predictions is not None:
                    # Extend generated tokens with province predictions
                    # This is a workaround to return both in the predictions tensor
                    predictions = torch.cat([
                        generated_tokens,
                        province_predictions.unsqueeze(-1)  # Add province as last "token"
                    ], dim=-1)
                else:
                    predictions = generated_tokens
                
                loss = None
                if has_labels:
                    # Calculate loss for evaluation metrics
                    with torch.no_grad():
                        outputs = model(
                            pixel_values=pixel_values,
                            labels=labels,
                            province_labels=province_labels,
                        )
                        loss = outputs.loss
            else:
                # Standard forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                    province_labels=province_labels,
                )
                loss = outputs.loss
                predictions = outputs.text_logits
        
        if prediction_loss_only:
            return loss, None, None
        
        # Prepare labels for return
        if has_labels:
            if has_province_labels:
                # Combine text labels and province labels
                combined_labels = torch.cat([
                    labels,
                    province_labels.unsqueeze(-1)
                ], dim=-1)
            else:
                combined_labels = labels
        else:
            combined_labels = None
        
        return loss, predictions, combined_labels


def create_multitask_data_collator(processor, province_label_key="province_code"):
    """
    Create a data collator that handles both text and province labels.
    """
    tokenizer = processor.tokenizer
    
    def collate_fn(features):
        # Extract pixel values from features
        pixel_values_list = [f["pixel_values"] for f in features]
        
        # Stack pixel values
        batch = {
            "pixel_values": torch.stack(pixel_values_list)
        }
        
        # Text labels
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            # Pad labels and validate token IDs
            max_length = max(len(label) for label in labels)
            padded_labels = []
            for label in labels:
                # Validate all token IDs are within vocabulary range
                validated_label = []
                current_vocab_size = len(tokenizer)  # Use len(tokenizer) for updated vocab size
                for token_id in label:
                    if 0 <= token_id < current_vocab_size:
                        validated_label.append(token_id)
                    else:
                        print(f"WARNING: Token ID {token_id} out of range [0, {current_vocab_size}), using pad token")
                        validated_label.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
                validated_tensor = torch.tensor(validated_label, dtype=torch.long)
                
                padded = torch.cat([
                    validated_tensor,
                    torch.full((max_length - len(validated_tensor),), -100, dtype=validated_tensor.dtype)
                ])
                padded_labels.append(padded)
            batch["labels"] = torch.stack(padded_labels)
        
        # Province labels
        if province_label_key in features[0]:
            province_codes = [f[province_label_key] for f in features]
            province_ids = [province_code_to_id(code) for code in province_codes]
            # Validate all province IDs are in valid range
            valid_province_ids = []
            for i, (code, pid) in enumerate(zip(province_codes, province_ids)):
                if pid < 0 or pid >= 77:  # 77 Thai provinces
                    print(f"WARNING: Invalid province ID {pid} for code '{code}' at index {i}, using 0")
                    valid_province_ids.append(0)
                else:
                    valid_province_ids.append(pid)
            batch["province_labels"] = torch.tensor(valid_province_ids, dtype=torch.long)
        
        return batch
    
    return collate_fn


def compute_multitask_metrics(processor, predict_province=True):
    """
    Create a metrics function for multi-task evaluation.
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        if predictions is None or labels is None:
            return {}
        
        # Split predictions and labels
        if predict_province and predictions.shape[-1] > labels.shape[-1]:
            # Last column contains province predictions
            text_predictions = predictions[:, :-1]
            province_predictions = predictions[:, -1].astype(int)
            
            text_labels = labels[:, :-1] if labels.shape[-1] > text_predictions.shape[-1] else labels
            province_labels = labels[:, -1].astype(int) if labels.shape[-1] > text_predictions.shape[-1] else None
        else:
            text_predictions = predictions
            text_labels = labels
            province_predictions = None
            province_labels = None
        
        # Decode text predictions and labels
        text_predictions = torch.from_numpy(text_predictions)
        text_labels = torch.from_numpy(text_labels)
        
        # Replace -100 with pad token id for decoding
        text_labels = torch.where(text_labels != -100, text_labels, processor.tokenizer.pad_token_id)
        text_predictions = torch.where(text_predictions != -100, text_predictions, processor.tokenizer.pad_token_id)
        
        pred_texts = processor.tokenizer.batch_decode(text_predictions, skip_special_tokens=True)
        label_texts = processor.tokenizer.batch_decode(text_labels, skip_special_tokens=True)
        
        # Text metrics (CER)
        from train_trocr import character_error_rate
        cer_scores = [character_error_rate(p.strip(), l.strip()) for p, l in zip(pred_texts, label_texts)]
        text_exact = sum(1 for p, l in zip(pred_texts, label_texts) if p.strip() == l.strip())
        
        metrics = {
            "text_cer": float(sum(cer_scores) / max(1, len(cer_scores))),
            "text_exact_match": float(text_exact / max(1, len(pred_texts))),
        }
        
        # Province metrics
        if predict_province and province_predictions is not None and province_labels is not None:
            province_correct = sum(1 for p, l in zip(province_predictions, province_labels) if p == l)
            metrics.update({
                "province_accuracy": float(province_correct / max(1, len(province_predictions))),
                "combined_exact_match": float(
                    sum(1 for pt, pl, pp, prov_p in zip(pred_texts, label_texts, province_predictions, province_labels)
                        if pt.strip() == pl.strip() and pp == prov_p) / max(1, len(pred_texts))
                ),
            })
        
        return metrics
    
    return compute_metrics