#!/usr/bin/env python3
"""
SmolVLM模型结构打印脚本
用于查看和了解SmolVLM的网络架构
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse
from collections import defaultdict
import json

def print_model_structure(model, max_depth=3, current_depth=0, module_name="", detailed=False):
    """
    递归打印模型结构
    
    Args:
        model: PyTorch模型
        max_depth: 最大打印深度
        current_depth: 当前深度
        module_name: 当前模块名称
        detailed: 是否打印详细信息
    """
    if current_depth >= max_depth:
        return
    
    indent = "  " * current_depth
    
    # 获取模块信息
    if hasattr(model, 'named_modules'):
        modules = list(model.named_modules())
    else:
        modules = [(module_name, model)]
    
    for name, module in modules:
        if current_depth == 0:
            full_name = name
        else:
            full_name = f"{module_name}.{name}" if module_name else name
        
        # 获取模块类型
        module_type = type(module).__name__
        
        # 获取参数数量
        param_count = sum(p.numel() for p in module.parameters())
        trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # 获取模块形状信息
        shape_info = ""
        if hasattr(module, 'weight') and module.weight is not None:
            shape_info = f" [weight: {list(module.weight.shape)}]"
        elif hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            shape_info = f" [{module.in_features} -> {module.out_features}]"
        
        # 打印模块信息
        print(f"{indent}{full_name} ({module_type}){shape_info}")
        
        if detailed and param_count > 0:
            print(f"{indent}  Parameters: {param_count:,} (trainable: {trainable_count:,})")
        
        # 递归打印子模块
        if hasattr(module, 'children') and current_depth < max_depth - 1:
            for child_name, child_module in module.named_children():
                print_model_structure(
                    child_module, 
                    max_depth, 
                    current_depth + 1, 
                    full_name,
                    detailed
                )

def analyze_model_architecture(model):
    """
    分析模型架构，统计各组件信息
    """
    print("\n" + "="*60)
    print("模型架构分析")
    print("="*60)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"参数压缩率: {(1 - trainable_params/total_params)*100:.2f}%")
    
    # 分析主要组件
    components = defaultdict(int)
    for name, module in model.named_modules():
        module_type = type(module).__name__
        components[module_type] += 1
    
    print(f"\n主要组件统计:")
    for comp_type, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comp_type}: {count}")
    
    # 分析视觉模型
    if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
        vision_model = model.model.vision_model
        vision_params = sum(p.numel() for p in vision_model.parameters())
        print(f"\n视觉模型参数数量: {vision_params:,}")
    
    # 分析语言模型
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        text_model = model.model.text_model
        text_params = sum(p.numel() for p in text_model.parameters())
        print(f"语言模型参数数量: {text_params:,}")
    
    # 分析连接器
    if hasattr(model, 'model') and hasattr(model.model, 'connector'):
        connector = model.model.connector
        connector_params = sum(p.numel() for p in connector.parameters())
        print(f"连接器参数数量: {connector_params:,}")

def main():
    parser = argparse.ArgumentParser(description="打印SmolVLM模型结构")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM-Instruct", 
                       help="模型ID或路径")
    parser.add_argument("--max_depth", type=int, default=4, 
                       help="最大打印深度")
    parser.add_argument("--detailed", action="store_true", 
                       help="是否打印详细信息")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="设备类型 (cpu/cuda)")
    parser.add_argument("--save_structure", type=str, default="", 
                       help="保存结构信息到JSON文件")
    
    args = parser.parse_args()
    
    print(f"正在加载模型: {args.model_id}")
    print(f"设备: {args.device}")
    print(f"最大深度: {args.max_depth}")
    print(f"详细信息: {args.detailed}")
    
    try:
        # 加载处理器和模型
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            device_map=args.device if args.device != "cpu" else None
        )
        
        print(f"\n模型加载成功!")
        print(f"模型类型: {type(model).__name__}")
        
        # 打印模型结构
        print("\n" + "="*60)
        print("模型结构")
        print("="*60)
        print_model_structure(model, args.max_depth, detailed=args.detailed)
        
        # 分析模型架构
        analyze_model_architecture(model)
        
        # 保存结构信息
        if args.save_structure:
            structure_info = {
                "model_id": args.model_id,
                "model_type": type(model).__name__,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "components": {}
            }
            
            # 收集主要组件信息
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # 叶子节点
                    module_type = type(module).__name__
                    if module_type not in structure_info["components"]:
                        structure_info["components"][module_type] = []
                    
                    param_info = {
                        "name": name,
                        "parameters": sum(p.numel() for p in module.parameters()),
                        "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)
                    }
                    structure_info["components"][module_type].append(param_info)
            
            with open(args.save_structure, 'w', encoding='utf-8') as f:
                json.dump(structure_info, f, indent=2, ensure_ascii=False)
            
            print(f"\n结构信息已保存到: {args.save_structure}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查模型ID是否正确，或者网络连接是否正常")

if __name__ == "__main__":
    main() 