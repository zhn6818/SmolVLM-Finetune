#!/usr/bin/env python3
"""
快速查看SmolVLM模型结构的简单脚本
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

def quick_view_model():
    """快速查看模型结构"""
    print("正在加载SmolVLM模型...")
    
    try:
        # 加载模型
        model_id = "HuggingFaceTB/SmolVLM-Instruct"
        model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float32)
        
        print(f"\n✅ 模型加载成功!")
        print(f"模型类型: {type(model).__name__}")
        
        # 打印主要组件
        print("\n" + "="*50)
        print("模型主要组件结构")
        print("="*50)
        
        if hasattr(model, 'model'):
            print("📁 model (主模型容器)")
            
            if hasattr(model.model, 'vision_model'):
                vision_model = model.model.vision_model
                vision_params = sum(p.numel() for p in vision_model.parameters())
                print(f"  👁️  vision_model (视觉模型) - 参数: {vision_params:,}")
                
                # 打印视觉模型的子组件
                for name, child in vision_model.named_children():
                    child_params = sum(p.numel() for p in child.parameters())
                    print(f"    ├── {name} - 参数: {child_params:,}")
            
            if hasattr(model.model, 'text_model'):
                text_model = model.model.text_model
                text_params = sum(p.numel() for p in text_model.parameters())
                print(f"  💬  text_model (语言模型) - 参数: {text_params:,}")
                
                # 打印语言模型的子组件
                for name, child in text_model.named_children():
                    child_params = sum(p.numel() for p in child.parameters())
                    print(f"    ├── {name} - 参数: {child_params:,}")
            
            if hasattr(model.model, 'connector'):
                connector = model.model.connector
                connector_params = sum(p.numel() for p in connector.parameters())
                print(f"  🔗  connector (连接器) - 参数: {connector_params:,}")
        
        if hasattr(model, 'lm_head'):
            lm_head = model.lm_head
            lm_params = sum(p.numel() for p in lm_head.parameters())
            print(f"  🎯  lm_head (语言模型头部) - 参数: {lm_params:,}")
        
        # 统计总参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📊 总参数数量: {total_params:,}")
        
        # 打印模型配置信息
        print(f"\n⚙️  模型配置信息:")
        print(f"  - 最大序列长度: {getattr(model.config, 'max_position_embeddings', 'N/A')}")
        print(f"  - 隐藏层维度: {getattr(model.config, 'hidden_size', 'N/A')}")
        print(f"  - 注意力头数: {getattr(model.config, 'num_attention_heads', 'N/A')}")
        print(f"  - 层数: {getattr(model.config, 'num_hidden_layers', 'N/A')}")
        
        print(f"\n🎉 模型结构查看完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查网络连接或模型ID是否正确")

if __name__ == "__main__":
    quick_view_model() 