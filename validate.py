import os
import sys
import argparse
import json
import warnings
import torch
from utils.utils_for_fast_inference import get_args, build_batch
from utils.build_model import build_model
from utils.volume import volume_computation3

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Validate GRAM Model with Different Tasks")
    parser.add_argument('--pretrain_dir', type=str, default='../方案1：预训练模型推理/pretrained_models/GRAM_pretrained_4modalities', 
                        help='Path to pretrained model directory')
    parser.add_argument('--task', type=str, default='ret%tva', 
                        choices=['ret%tva', 'ret%tv', 'ret%ta'], 
                        help='Task type: ret%tva (text-video-audio), ret%tv (text-video), ret%ta (text-audio)')
    parser.add_argument('--text', type=str, nargs='+', 
                        default=['A dog is barking', 'A dog is howling'], 
                        help='Text queries for validation')
    parser.add_argument('--video', type=str, nargs='+', 
                        default=['./assets/videos/video1.mp4', './assets/videos/video2.mp4'], 
                        help='Video paths for validation')
    parser.add_argument('--audio', type=str, nargs='+', 
                        default=['./assets/audios/audio1.mp3', './assets/audios/audio2.mp3'], 
                        help='Audio paths for validation')
    parser.add_argument('--output_dir', type=str, default='output/validation', 
                        help='Output directory for validation results')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], 
                        help='Device to run validation on')
    
    args = parser.parse_args()
    
    # Set environment variable for local rank
    os.environ['LOCAL_RANK'] = '0'
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.pretrain_dir}...")
    model_args = get_args(args.pretrain_dir)
    model, _, _ = build_model(model_args)
    model.to(args.device)
    model.eval()
    
    # Build batch
    print("Building batch...")
    batch = build_batch(model_args, args.text, args.video, args.audio)
    if batch is None:
        print("Error: Failed to build batch. Please check your input paths.")
        return
    
    # Run validation
    print(f"Running validation with task: {args.task}...")
    with torch.no_grad():
        evaluation_dict = model(batch, args.task, compute_loss=False)
    
    # Compute volume if all three modalities are used
    if args.task == 'ret%tva':
        feat_t = evaluation_dict['feat_t']
        feat_v = evaluation_dict['feat_v']
        feat_a = evaluation_dict['feat_a']
        volume = volume_computation3(feat_t, feat_v, feat_a)
        print(f"Volume: {volume.detach().cpu()}")
        
        # Save results
        results = {
            'task': args.task,
            'text': args.text,
            'video': args.video,
            'audio': args.audio,
            'volume': volume.detach().cpu().tolist()
        }
    elif args.task == 'ret%tv':
        feat_t = evaluation_dict['feat_t']
        feat_v = evaluation_dict['feat_v']
        print(f"Text features shape: {feat_t.shape}")
        print(f"Video features shape: {feat_v.shape}")
        
        # Save results
        results = {
            'task': args.task,
            'text': args.text,
            'video': args.video,
            'text_features_shape': feat_t.shape,
            'video_features_shape': feat_v.shape
        }
    elif args.task == 'ret%ta':
        feat_t = evaluation_dict['feat_t']
        feat_a = evaluation_dict['feat_a']
        print(f"Text features shape: {feat_t.shape}")
        print(f"Audio features shape: {feat_a.shape}")
        
        # Save results
        results = {
            'task': args.task,
            'text': args.text,
            'audio': args.audio,
            'text_features_shape': feat_t.shape,
            'audio_features_shape': feat_a.shape
        }
    
    # Save results to JSON file
    results_file = os.path.join(args.output_dir, f'validation_results_{args.task}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Validation results saved to {results_file}")

if __name__ == '__main__':
    main()