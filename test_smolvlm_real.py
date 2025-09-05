#!/usr/bin/env python3
"""
SmolVLM真实图像测试 - 使用网络图片测试模型性能
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

def load_real_image():
    """加载真实的网络图片"""
    # 使用一个公开的示例图片
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except:
        # 如果网络图片加载失败，创建一个更复杂的本地图片
        image = Image.new('RGB', (500, 400), color='white')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        # 绘制一个房子
        draw.rectangle([100, 200, 300, 350], fill='brown', outline='black', width=3)
        draw.polygon([(80, 200), (200, 120), (320, 200)], fill='red', outline='black', width=3)
        draw.rectangle([150, 250, 200, 300], fill='blue', outline='black', width=2)
        draw.rectangle([220, 250, 270, 300], fill='blue', outline='black', width=2)
        
        # 绘制太阳
        draw.ellipse([350, 50, 400, 100], fill='yellow', outline='orange', width=2)
        
        # 绘制云朵
        draw.ellipse([50, 50, 120, 80], fill='white', outline='gray', width=1)
        draw.ellipse([80, 40, 150, 70], fill='white', outline='gray', width=1)
        
        return image

def test_smolvlm_real():
    """测试SmolVLM在真实图像上的性能"""
    print("🚀 SmolVLM真实图像测试")
    print("="*60)
    
    try:
        # 加载模型
        print("📥 正在加载模型...")
        model_id = "HuggingFaceTB/SmolVLM-Instruct"
        model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(model_id)
        
        print("✅ 模型加载成功!")
        
        # 加载真实图片
        print("\n📸 加载真实图片...")
        image = load_real_image()
        
        print(f"图片尺寸: {image.size}")
        
        # 测试1: 图像描述
        print("\n" + "="*60)
        print("测试1: 图像描述")
        print("="*60)
        
        prompt1 = "<image>\nDescribe what you see in this image in detail."
        print(f"输入: {prompt1}")
        
        inputs1 = processor(text=prompt1, images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs1 = model.generate(
                **inputs1, 
                max_new_tokens=150, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        response1 = processor.decode(outputs1[0], skip_special_tokens=True)
        print(f"输出: {response1}")
        
        # 测试2: 视觉问答
        print("\n" + "="*60)
        print("测试2: 视觉问答")
        print("="*60)
        
        prompt2 = "<image>\nWhat colors do you see in this image?"
        print(f"输入: {prompt2}")
        
        inputs2 = processor(text=prompt2, images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs2 = model.generate(
                **inputs2, 
                max_new_tokens=100, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        response2 = processor.decode(outputs2[0], skip_special_tokens=True)
        print(f"输出: {response2}")
        
        # 测试3: 物体识别
        print("\n" + "="*60)
        print("测试3: 物体识别")
        print("="*60)
        
        prompt3 = "<image>\nWhat objects can you identify in this image?"
        print(f"输入: {prompt3}")
        
        inputs3 = processor(text=prompt3, images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs3 = model.generate(
                **inputs3, 
                max_new_tokens=100, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        response3 = processor.decode(outputs3[0], skip_special_tokens=True)
        print(f"输出: {response3}")
        
        print("\n🎉 测试完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_smolvlm_real()
