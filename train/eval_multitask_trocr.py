"""
Evaluation script for multi-task TrOCR with image-based province prediction.

This script evaluates a trained multi-task model on both:
1. License plate text recognition accuracy
2. Province classification accuracy (image-based)

Usage:
    python eval_multitask_trocr.py --model-path outputs/multitask/ --csv data.csv --data-root images/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import TrOCRProcessor
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from multi_task_trocr_fixed import (
    MultiTaskTrOCR,
    province_code_to_id,
    THAI_PROVINCES
)

def id_to_province_code(province_id: int) -> str:
    """Convert province ID back to province code."""
    if 0 <= province_id < len(THAI_PROVINCES):
        return THAI_PROVINCES[province_id]
    return "TH-10"  # Default to Bangkok
from train_trocr import (
    load_records,
    stratified_split,
    character_error_rate,
)

logger = logging.getLogger(__name__)


def evaluate_multitask_model(
    model: MultiTaskTrOCR,
    processor: TrOCRProcessor,
    records: List,
    batch_size: int = 4,
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Evaluate multi-task model on license plate recognition and province classification.
    """
    model.eval()
    device = next(model.parameters()).device
    
    if max_samples:
        records = records[:max_samples]
    
    results = []
    text_predictions = []
    text_labels = []
    province_predictions = []
    province_labels = []
    
    # Process in batches
    total_time = 0
    num_batches = (len(records) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(records))
            batch_records = records[start_idx:end_idx]
            
            # Prepare batch
            images = []
            batch_text_labels = []
            batch_province_labels = []
            
            for record in batch_records:
                from train.data_utils import load_plate_crop
                image = load_plate_crop(record)
                images.append(image)
                batch_text_labels.append(record.plate_text)
                batch_province_labels.append(record.province_code)
            
            # Process images
            start_time = time.time()
            encoding = processor(images=images, return_tensors="pt")
            pixel_values = encoding.pixel_values.to(device)
            
            # Generate predictions with stopping criteria for plate text
            generation_outputs = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=15,  # Reduced to focus on plate text only
                num_beams=3,
                do_sample=False,
                early_stopping=True,
                # Add stopping criteria if available
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            generated_ids = generation_outputs['sequences']
            province_preds = generation_outputs['province_predictions']
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Decode text predictions
            pred_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Store predictions and labels
            for i, (record, pred_text, province_pred) in enumerate(zip(batch_records, pred_texts, province_preds)):
                # Clean predicted text (remove province tokens and unwanted text)
                # 1. Remove province tokens if present
                if "<prov>" in pred_text:
                    pred_text = pred_text.split("<prov>")[0].strip()
                
                # 2. Remove common unwanted patterns that appear after plate text
                unwanted_patterns = [
                    "พิธีกรรายการ",
                    "รายการ", 
                    "__",
                    "___",
                    "Commonwealth",
                    "ตลอดผ่าตัด",
                    "หนองบัวได้",
                    "ตลอดเสริม"
                ]
                
                # Split by spaces and filter out unwanted parts
                words = pred_text.split()
                cleaned_words = []
                
                for word in words:
                    # Skip if word is in unwanted patterns
                    if any(pattern in word for pattern in unwanted_patterns):
                        break  # Stop at first unwanted pattern
                    cleaned_words.append(word)
                
                pred_text = " ".join(cleaned_words).strip()
                
                # 3. Additional cleaning: extract only plate-like text (Thai characters + numbers)
                import re
                # Match Thai license plate pattern: Thai chars + numbers, more precise
                # Thai plates typically: [digit][Thai char][Thai char][digit digit digit digit] or similar
                plate_pattern = r'^([0-9]*[ก-๙]+[0-9]*[ก-๙]*[0-9]*)'
                
                # Remove spaces first
                clean_text = pred_text.replace(" ", "")
                match = re.match(plate_pattern, clean_text)
                if match and len(match.group(1)) >= 3:  # At least 3 characters for a valid plate
                    pred_text = match.group(1)
                elif len(clean_text) <= 10 and re.match(r'^[ก-๙0-9]+$', clean_text):
                    # If it's short and only contains valid plate characters
                    pred_text = clean_text
                else:
                    # Fallback: take only first part before unwanted text
                    words = pred_text.split()
                    if words:
                        first_word = words[0].replace(" ", "")
                        if re.match(r'^[ก-๙0-9]+$', first_word):
                            pred_text = first_word
                        else:
                            pred_text = pred_text  # Keep original if nothing else works
                
                province_code = id_to_province_code(province_pred.item())
                
                result = {
                    "image_name": str(record.plate_image_path.name),
                    "ground_truth_text": record.plate_text,
                    "predicted_text": pred_text,
                    "ground_truth_province": record.province_code,
                    "predicted_province": province_code,
                    "text_cer": character_error_rate(pred_text, record.plate_text),
                    "text_exact_match": pred_text.strip() == record.plate_text.strip(),
                    "province_correct": province_code == record.province_code,
                }
                results.append(result)
                
                text_predictions.append(pred_text)
                text_labels.append(record.plate_text)
                province_predictions.append(province_code)
                province_labels.append(record.province_code)
    
    # Calculate overall metrics
    text_cer = sum(r["text_cer"] for r in results) / len(results)
    text_exact_match = sum(r["text_exact_match"] for r in results) / len(results)
    province_accuracy = sum(r["province_correct"] for r in results) / len(results)
    
    # Combined exact match (both text and province correct)
    combined_exact = sum(
        r["text_exact_match"] and r["province_correct"] for r in results
    ) / len(results)
    
    # Performance metrics
    avg_latency_ms = (total_time / len(records)) * 1000
    throughput_ips = len(records) / total_time
    
    metrics = {
        "samples": len(results),
        "text_cer": text_cer,
        "text_exact_match": text_exact_match,
        "province_accuracy": province_accuracy,
        "combined_exact_match": combined_exact,
        "latency_ms": avg_latency_ms,
        "throughput_ips": throughput_ips,
    }
    
    return {
        "metrics": metrics,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-task TrOCR model")
    
    # Model arguments
    parser.add_argument("--model-path", required=True, help="Path to trained model directory")
    parser.add_argument("--model-id", default="openthaigpt/thai-trocr", help="Base model ID")
    
    # Data arguments
    parser.add_argument("--csv", required=True, help="Path to CSV file with test data")
    parser.add_argument("--data-root", required=True, help="Root directory for images")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Data split to evaluate")
    
    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    
    # Output arguments
    parser.add_argument("--save-results", help="Path to save detailed results JSON")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    # Load data
    logger.info(f"Loading data from {args.csv}")
    records = load_records(args.csv, data_roots=[args.data_root])
    
    # Split data
    splits = stratified_split(records, seed=42)
    test_records = splits[args.split]
    
    logger.info(f"Evaluating on {len(test_records)} {args.split} samples")
    
    # Load processor
    processor = TrOCRProcessor.from_pretrained(args.model_path)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    try:
        # Try loading the saved model first
        model = MultiTaskTrOCR.from_pretrained(args.model_path)
        logger.info("Successfully loaded model from pretrained")
    except Exception as e:
        logger.warning(f"Failed to load pretrained model: {e}")
        
        # Fallback: Load full model state dict if available
        import os
        state_dict_path = os.path.join(args.model_path, "full_model_state.pt")
        if os.path.exists(state_dict_path):
            logger.info(f"Trying to load from state dict: {state_dict_path}")
            from train.multi_task_trocr_fixed import create_multitask_model
            model = create_multitask_model(args.model_id, num_provinces=77)
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded model from state dict")
        else:
            # Last resort: Create fresh model
            logger.warning("Creating fresh model - this will have poor performance")
            from train.multi_task_trocr_fixed import create_multitask_model
            model = create_multitask_model(args.model_id, num_provinces=77)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Evaluate
    logger.info("Starting evaluation")
    eval_results = evaluate_multitask_model(
        model=model,
        processor=processor,
        records=test_records,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    
    # Print results
    metrics = eval_results["metrics"]
    logger.info("Evaluation Results:")
    logger.info(f"  Samples: {metrics['samples']}")
    logger.info(f"  Text CER: {metrics['text_cer']:.4f}")
    logger.info(f"  Text Exact Match: {metrics['text_exact_match']:.4f}")
    logger.info(f"  Province Accuracy: {metrics['province_accuracy']:.4f}")
    logger.info(f"  Combined Exact Match: {metrics['combined_exact_match']:.4f}")
    logger.info(f"  Latency: {metrics['latency_ms']:.2f} ms")
    logger.info(f"  Throughput: {metrics['throughput_ips']:.2f} images/sec")
    
    # Save detailed results
    if args.save_results:
        logger.info(f"Saving detailed results to {args.save_results}")
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    # Print sample predictions
    logger.info("\nSample predictions:")
    for i, result in enumerate(eval_results["results"][:5]):
        logger.info(f"  Sample {i+1}:")
        logger.info(f"    Ground truth: '{result['ground_truth_text']}' | {result['ground_truth_province']}")
        logger.info(f"    Prediction:   '{result['predicted_text']}' | {result['predicted_province']}")
        logger.info(f"    Text CER: {result['text_cer']:.3f} | Province: {'✓' if result['province_correct'] else '✗'}")


if __name__ == "__main__":
    main()