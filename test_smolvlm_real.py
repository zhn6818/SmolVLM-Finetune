#!/usr/bin/env python3
"""
SmolVLM真实图像测试 - 使用网络图片测试模型性能
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import re

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

def smart_generate(model, processor, inputs, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """
    智能生成函数，实现更优雅的终止
    """
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            # 修复warning：num_beams=1时不用early_stopping
            early_stopping=False,
            num_beams=1,
            no_repeat_ngram_size=3,
            # 设置最小长度确保有足够内容
            min_new_tokens=10,
            # 使用更智能的停止策略
            use_cache=True
        )
    
    # 解码输出
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 后处理：清理和优化输出
    response = clean_response(response, inputs, processor)
    
    return response

def clean_response(response, inputs=None, processor=None):
    """
    清理和优化模型输出，移除重复的输入内容
    """
    # 获取原始输入文本
    if inputs is not None and 'input_ids' in inputs and processor is not None:
        input_text = processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
        # 移除<image>标记后的内容作为输入部分
        if "<image>" in input_text:
            input_parts = input_text.split("<image>")
            if len(input_parts) > 1:
                input_question = input_parts[-1].strip()
            else:
                input_question = input_text
        else:
            input_question = input_text
    else:
        input_question = ""
    
    # 移除输入部分（如果存在）
    if "<image>" in response:
        # 找到最后一个<image>后的内容
        parts = response.split("<image>")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    # 移除重复的输入问题
    if input_question and input_question in response:
        response = response.replace(input_question, "").strip()
    
    # 移除重复的句子
    sentences = response.split('.')
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen and len(sentence) > 5:  # 过滤太短的句子
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    # 重新组合
    response = '. '.join(unique_sentences)
    
    # 确保以句号结尾
    if response and not response.endswith(('.', '!', '?')):
        response += '.'
    
    # 移除多余的空格
    response = re.sub(r'\s+', ' ', response).strip()
    
    return response

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
        
        # 使用智能生成函数
        response1 = smart_generate(model, processor, inputs1, max_new_tokens=200, temperature=0.7, top_p=0.9)
        print(f"输出: {response1}")
        
        # 测试2: 视觉问答
        print("\n" + "="*60)
        print("测试2: 视觉问答")
        print("="*60)
        
        prompt2 = "<image>\nWhat colors do you see in this image?"
        print(f"输入: {prompt2}")
        
        inputs2 = processor(text=prompt2, images=image, return_tensors="pt")
        
        # 使用智能生成函数
        response2 = smart_generate(model, processor, inputs2, max_new_tokens=150, temperature=0.7, top_p=0.9)
        print(f"输出: {response2}")
        
        # 测试3: 物体识别
        print("\n" + "="*60)
        print("测试3: 物体识别")
        print("="*60)
        
        prompt3 = "<image>\nWhat objects can you identify in this image?"
        print(f"输入: {prompt3}")
        
        inputs3 = processor(text=prompt3, images=image, return_tensors="pt")
        
        # 使用智能生成函数
        response3 = smart_generate(model, processor, inputs3, max_new_tokens=150, temperature=0.7, top_p=0.9)
        print(f"输出: {response3}")
        
        # 测试4: 对比不同生成策略
        print("\n" + "="*60)
        print("测试4: 对比不同生成策略")
        print("="*60)
        
        prompt4 = "<image>\nWrite a short story about what you see in this image."
        print(f"输入: {prompt4}")
        
        inputs4 = processor(text=prompt4, images=image, return_tensors="pt")
        
        # 策略1: 保守生成（低温度）
        print("\n📝 策略1: 保守生成 (temperature=0.3)")
        response4a = smart_generate(model, processor, inputs4, max_new_tokens=200, temperature=0.3, top_p=0.8)
        print(f"输出: {response4a}")
        
        # 策略2: 创意生成（高温度）
        print("\n🎨 策略2: 创意生成 (temperature=0.9)")
        response4b = smart_generate(model, processor, inputs4, max_new_tokens=200, temperature=0.9, top_p=0.95)
        print(f"输出: {response4b}")
        
        print("\n🎉 测试完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_smolvlm_real()
