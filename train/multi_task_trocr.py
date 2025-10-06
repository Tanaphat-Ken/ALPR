"""
Multi-task TrOCR model for license plate recognition wi        # Province classifier head
        self.province_classifier = nn.Sequential(
            nn.Linear(encoder_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_provinces)
        )
        
        # Projection layer to match encoder-decoder hidden sizes
        if encoder_hidden_size != decoder_hidden_size:
            self.encoder_decoder_projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.encoder_decoder_projection = None
        
        # Loss weights
        self.text_loss_weight = 1.0
        self.province_loss_weight = 0.5
        
        # Initialize province classifier
        self._init_province_classifier()ovince prediction.

This module implements a TrOCR variant that performs both:
1. Text recognition (original TrOCR task)
2. Province classification from visual features (new auxiliary task)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import VisionEncoderDecoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput


class MultiTaskTrOCROutput:
    """Output class for multi-task TrOCR model."""
    
    def __init__(
        self,
        text_loss: Optional[torch.Tensor] = None,
        province_loss: Optional[torch.Tensor] = None,
        text_logits: Optional[torch.Tensor] = None,
        province_logits: Optional[torch.Tensor] = None,
        encoder_last_hidden_state: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.text_loss = text_loss
        self.province_loss = province_loss
        self.text_logits = text_logits
        self.province_logits = province_logits
        self.encoder_last_hidden_state = encoder_last_hidden_state
        
        # For compatibility with Seq2SeqLMOutput
        self.loss = text_loss  # Primary loss
        self.logits = text_logits  # Primary logits
        
        # Store additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class MultiTaskTrOCR(VisionEncoderDecoderModel):
    """
    Multi-task TrOCR model that performs both text recognition and province classification.
    
    The model uses:
    - Vision encoder to extract image features
    - Text decoder for license plate text generation  
    - Province classifier head for direct image-based province prediction
    """
    
    def __init__(self, config=None, encoder=None, decoder=None, num_provinces=77):
        super().__init__(config, encoder, decoder)
        
        self.num_provinces = num_provinces
        
        # Get hidden sizes
        encoder_hidden_size = self.encoder.config.hidden_size
        decoder_hidden_size = self.decoder.config.hidden_size
        
        # Province classification head
        self.province_classifier = nn.Sequential(
            nn.Linear(encoder_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_provinces)
        )
        
        # Projection layer to match encoder-decoder hidden sizes
        if encoder_hidden_size != decoder_hidden_size:
            self.encoder_decoder_projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.encoder_decoder_projection = None
        
        # Loss weights
        self.text_loss_weight = 1.0
        self.province_loss_weight = 0.5
        
        # Initialize province classifier
        self._init_province_classifier()
    
    def _init_province_classifier(self):
        """Initialize province classifier weights."""
        for module in self.province_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _shift_right(self, input_ids):
        """Shift input ids one token to the right for teacher forcing."""
        if input_ids is None:
            return None
            
        # Get start token id
        start_token_id = getattr(self.config, 'decoder_start_token_id', None) or \
                        getattr(self.config, 'bos_token_id', None) or 0
        
        # Create shifted input
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        
        return shifted_input_ids
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values: Optional[Tuple] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        province_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> MultiTaskTrOCROutput:
        """
        Forward pass for multi-task TrOCR.
        
        Args:
            pixel_values: Input images
            labels: Text labels for decoder
            province_labels: Province class labels (0-76)
            **kwargs: Other arguments for base model
            
        Returns:
            MultiTaskTrOCROutput with text and province predictions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get encoder outputs
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        
        # Project encoder hidden states if needed for cross-attention
        projected_encoder_states = encoder_hidden_states
        if self.encoder_decoder_projection is not None:
            projected_encoder_states = self.encoder_decoder_projection(encoder_hidden_states)
        
        # Text generation (original TrOCR task)
        # Prepare decoder inputs properly
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if labels is not None:
                # Create decoder_input_ids by shifting labels right for teacher forcing
                decoder_input_ids = self._shift_right(labels)
            else:
                # Create default start tokens for generation
                batch_size = encoder_hidden_states.shape[0]
                start_token_id = getattr(self.config, 'decoder_start_token_id', None) or \
                               getattr(self.config, 'bos_token_id', None) or 0
                decoder_input_ids = torch.full(
                    (batch_size, 1), 
                    start_token_id, 
                    dtype=torch.long, 
                    device=encoder_hidden_states.device
                )
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=projected_encoder_states,
            encoder_attention_mask=None,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        
        text_logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        
        # Province classification from encoder features
        # Use global average pooling over spatial dimensions
        # encoder_hidden_states shape: [batch_size, seq_len, hidden_size]
        pooled_features = encoder_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        province_logits = self.province_classifier(pooled_features)
        
        # Calculate losses
        text_loss = None
        province_loss = None
        total_loss = None
        
        if labels is not None:
            # Text loss (cross-entropy)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            text_loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), labels.view(-1))
            total_loss = self.text_loss_weight * text_loss
        
        if province_labels is not None:
            # Province loss (cross-entropy)
            loss_fct = nn.CrossEntropyLoss()
            province_loss = loss_fct(province_logits, province_labels)
            
            if total_loss is not None:
                total_loss += self.province_loss_weight * province_loss
            else:
                total_loss = self.province_loss_weight * province_loss
        
        return MultiTaskTrOCROutput(
            text_loss=text_loss,
            province_loss=province_loss,
            text_logits=text_logits,
            province_logits=province_logits,
            encoder_last_hidden_state=encoder_hidden_states,
            loss=total_loss,  # Combined loss for training
            logits=text_logits,  # Primary logits for compatibility
        )
    
    def generate(self, pixel_values, **kwargs):
        """
        Generate text while also predicting province.
        
        Returns both generated text and province predictions.
        """
        # Get encoder outputs for province prediction
        encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Province prediction
        pooled_features = encoder_hidden_states.mean(dim=1)
        province_logits = self.province_classifier(pooled_features)
        province_predictions = torch.argmax(province_logits, dim=-1)
        
        # Text generation (use parent's generate method)
        generated_ids = super().generate(
            pixel_values=pixel_values,
            encoder_outputs=encoder_outputs,
            **kwargs
        )
        
        return {
            'generated_ids': generated_ids,
            'province_predictions': province_predictions,
            'province_logits': province_logits,
        }


def create_multitask_trocr(base_model_id: str, num_provinces: int = 77) -> MultiTaskTrOCR:
    """
    Create a multi-task TrOCR model from a base TrOCR checkpoint.
    
    Args:
        base_model_id: Hugging Face model ID (e.g., "openthaigpt/thai-trocr")
        num_provinces: Number of province classes (default: 77 Thai provinces)
        
    Returns:
        MultiTaskTrOCR model
    """
    # Load base model
    base_model = VisionEncoderDecoderModel.from_pretrained(base_model_id)
    
    # Create multi-task model with same config
    model = MultiTaskTrOCR(
        config=base_model.config,
        encoder=base_model.encoder,
        decoder=base_model.decoder,
        num_provinces=num_provinces
    )
    
    return model


# Province mapping utilities
THAI_PROVINCES = [
    "TH-10", "TH-11", "TH-12", "TH-13", "TH-14", "TH-15", "TH-16", "TH-17", "TH-18", "TH-19",
    "TH-20", "TH-21", "TH-22", "TH-23", "TH-24", "TH-25", "TH-26", "TH-27", "TH-30", "TH-31",
    "TH-32", "TH-33", "TH-34", "TH-35", "TH-36", "TH-37", "TH-38", "TH-39", "TH-40", "TH-41",
    "TH-42", "TH-43", "TH-44", "TH-45", "TH-46", "TH-47", "TH-48", "TH-49", "TH-50", "TH-51",
    "TH-52", "TH-53", "TH-54", "TH-55", "TH-56", "TH-57", "TH-58", "TH-60", "TH-61", "TH-62",
    "TH-63", "TH-64", "TH-65", "TH-66", "TH-67", "TH-70", "TH-71", "TH-72", "TH-73", "TH-74",
    "TH-75", "TH-76", "TH-77", "TH-80", "TH-81", "TH-82", "TH-83", "TH-84", "TH-85", "TH-86",
    "TH-90", "TH-91", "TH-92", "TH-93", "TH-94", "TH-95", "TH-96"
]

PROVINCE_TO_ID = {province: i for i, province in enumerate(THAI_PROVINCES)}
ID_TO_PROVINCE = {i: province for i, province in enumerate(THAI_PROVINCES)}


def province_code_to_id(province_code: str) -> int:
    """Convert province code (e.g., 'TH-10') to class ID (0-76)."""
    return PROVINCE_TO_ID.get(province_code, -1)


def id_to_province_code(class_id: int) -> str:
    """Convert class ID (0-76) to province code (e.g., 'TH-10')."""
    return ID_TO_PROVINCE.get(class_id, "TH-10")  # Default to Bangkok