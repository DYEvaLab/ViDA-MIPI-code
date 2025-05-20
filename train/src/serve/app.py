import os
import sys
# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/train/src/serve
src_dir = os.path.dirname(current_dir)                    # /path/to/train/src
train_dir = os.path.dirname(src_dir)                      # /path/to/train
project_dir = os.path.dirname(train_dir)                  # /path/to
sys.path.insert(0, project_dir)
sys.path.insert(0, train_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

# Disable proxy settings that might cause HTTPX parsing errors
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["no_proxy"] = "*"

import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from train.src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# System prompt examples
SYSTEM_PROMPT_EXAMPLES = {
    "Default": "You are a helpful assistant.",
    "Overall Quality Analysis": "You are an expert in image quality assessment. Your task is to assess the overall quality of the provided image.\nTo assess the image quality, you should think step by step.\nFirst step, provide a brief description of the image content.\nSecond step, find distortions and analyze their impact on specific regions.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type. Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image. Find each distortion with grounding, which means in the response you should directly output the bounding box coordinates after identifying the distortions. The bounding box coordinates should be in the format (x1,y1),(x2,y2), where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner. All coordinates are normalized to the range of 0 to 1000.\nThird step, analyze the overall image quality and visual perception. If distortions are present, identify the key distortions that have the most significant impact on the overall image quality. Provide detailed reasoning about how these key distortions affect the image's overall visual perception, especially regarding low-level features like sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\nFinal step, conclude your answer with this sentence: Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent).\nSeparate each step with a line break.",
    "Individual Distortion Assessment": "You are an expert in image quality assessment. Your task is to identify and assess a specific distortion of the image in detail.\nIf you do not detect the specified distortion, simply state that this distortion is not present in the image.\nTo assess the distortion in detail, you need to think step by step.\n First step, provide the number of occurrences of distortion in the image.\n Second step, analyze each distortion in detail: first, describe its position; then, assess its visual manifestation and perceptual impact; and finally, evaluate how the distortion affects the image by considering the quality of the affected region, its importance, and the image's low-level attributes. Find each distortion with grounding, which means in the response you should directly output the bounding box coordinates after identifying the distortions. The bounding box coordinates should be in the format (x1,y1),(x2,y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. All coordinates are normalized to the range of 0 to 1000.\nSeparate each distortion explanation with a line break.\n Final step, summarize the impact of this distortion on the overall quality of the image.\nSeparate each step with a line break.",
    "Brief Assessment": "You are an expert in image quality assessment. Your task is to assess the overall quality of an image and give a structured and concise summary.\n- **Conclusion**: The quality level of the image (one of the following quality levels: 'bad', 'poor', 'fair', 'good', 'excellent').\n- **Distortion assessment**: List and describe the distortions in the image with brief explanations of their impact.\n  - Your analysis for each distortion should start with the number of occurrences of this distortion in the image.\n  - For each distortion, analyze the specific effect and how it affects the image.\n  - If there is no distortion in the image, summarize the detailed image quality assessment, especially low-level attributes (e.g., sharpness, clarity, contrast, lighting).\n- **Scene description**: A concise description of the scene in the image, focusing on the key elements.",
    "Other": "You are an expert in image quality assessment."
}

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def bot_streaming(message, history, system_prompt, generation_args):
    # Ensure system_prompt is a string
    if hasattr(system_prompt, 'value'):
        system_prompt = system_prompt.value
    
    # Initialize variables
    images = []
    videos = []

    # 处理上传的文件
    if isinstance(message, dict) and message.get("files"):
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item.get("path", "")
            elif isinstance(file_item, str):
                file_path = file_item
            else:
                file_path = str(file_item)
                
            if file_path and os.path.exists(file_path):
                if is_video_file(file_path):
                    videos.append(file_path)
                else:
                    images.append(file_path)
    elif isinstance(message, list):
        # 处理ChatInterface传入的格式，可能是列表
        for item in message:
            if isinstance(item, str):
                # 纯文本部分，不处理
                continue
            elif item and hasattr(item, "name"):
                file_path = item.name
                if file_path and os.path.exists(file_path):
                    if is_video_file(file_path):
                        videos.append(file_path)
                    else:
                        images.append(file_path)
    
    # 从消息中提取文本
    user_text = ""
    if isinstance(message, str):
        user_text = message
    elif isinstance(message, dict) and "text" in message:
        user_text = message["text"]
    elif isinstance(message, list):
        # 从列表中提取文本部分
        for item in message:
            if isinstance(item, str):
                user_text = item
                break

    conversation = []
        
    # 处理历史对话
    for user_turn, assistant_turn in history:
        user_content = []
        # 处理用户消息是元组的情况 (通常表示 (文件路径, 文本消息))
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            user_text_hist = user_turn[1] if len(user_turn) > 1 else ""
            
            # 确保file_paths是列表
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
                
            # 处理每个文件
            for file_path in file_paths:
                if file_path:  # 确保路径不为空
                    if is_video_file(file_path):
                        user_content.append({"type": "video", "video": file_path, "fps":1.0})
                    else:
                        user_content.append({"type": "image", "image": file_path})
                        
            # 添加文本内容
            if user_text_hist:
                user_content.append({"type": "text", "text": user_text_hist})
        else:
            # 如果只是普通文本
            user_content.append({"type": "text", "text": user_turn})
            
        # 添加到对话中
        conversation.append({"role": "user", "content": user_content})

        # 添加助手回复
        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})

    # 处理当前用户输入
    user_content = []
    # 添加图片
    for image in images:
        user_content.append({"type": "image", "image": image})
    # 添加视频
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps":1.0})
    # 添加文本
    if user_text:
        user_content.append({"type": "text", "text": user_text})
        
    # 将当前用户输入添加到对话中
    conversation.append({"role": "user", "content": user_content})

    # 始终替换系统提示词
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", f"<|im_start|>system\n{system_prompt}<|im_end|>")
    print(prompt)
    
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device) 

    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 为了与ChatInterface兼容，不能使用yield，而是收集所有输出
    buffer = ""
    for new_text in streamer:
        buffer += new_text
    
    # 返回字符串而不是通过yield生成
    return buffer

def update_system_prompt(example_name):
    """Return the system prompt based on the selected example name"""
    return SYSTEM_PROMPT_EXAMPLES.get(example_name, SYSTEM_PROMPT_EXAMPLES["Default"])

def apply_system_prompt(prompt_value, chatbot):
    """Apply system prompt without restarting conversation
    
    Args:
        prompt_value: Prompt content
        chatbot: Chat interface component
        
    Returns:
        Unchanged chat history, unchanged input, updated active prompt, and success message
    """
    # 只更新系统提示词，不重置对话历史
    print(f"Apply system prompt: {prompt_value[:80]}...")
    success_message = f"✅ System prompt updated: {prompt_value[:120]}..." if len(prompt_value) > 120 else f"✅ System prompt updated: {prompt_value}"
    return chatbot, "", prompt_value, success_message
        
def main(args):

    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_base = args.model_base, model_path = args.model_path, 
                                                device_map=args.device, model_name=model_name, 
                                                load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                                device=args.device, use_flash_attn=use_flash_attn
    )

    chatbot = gr.Chatbot(scale=2)
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image", "video"], placeholder="Enter message or upload file...",
                                  show_label=False)
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    with gr.Blocks(fill_height=True) as demo:
        with gr.Row():
            gr.HTML("<h1>" + model_name + "</h1>")
        
        # 创建状态变量
        active_system_prompt = gr.State(SYSTEM_PROMPT_EXAMPLES["Default"])
        
        with gr.Row():
            with gr.Column(scale=3):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter system prompt to define the AI assistant's behavior...",
                    value=SYSTEM_PROMPT_EXAMPLES["Default"],
                    lines=3
                )
                status_message = gr.Markdown("") # 添加状态消息显示
            
            with gr.Column(scale=1):
                with gr.Row():
                    system_prompt_examples = gr.Dropdown(
                        label="Prompt Templates",
                        choices=list(SYSTEM_PROMPT_EXAMPLES.keys()),
                        value="Default"
                    )
                
                with gr.Row():
                    gr.Markdown("**Apply system prompt:**")
                
                with gr.Row():
                    gr.Markdown("""
                    Applying a system prompt will affect future responses.
                    The prompt defines how the AI assistant behaves.
                    """)
                    
                with gr.Row():
                    gr.Markdown("**Please clear the chat before applying the system prompt**", elem_classes="warning-text")
                    
                with gr.Row():
                    apply_button = gr.Button("Apply prompt")
        
        # 创建聊天界面
        chat_interface = gr.ChatInterface(
            fn=lambda message, history, active_prompt: bot_streaming(message, history, active_prompt, generation_args),
            stop_btn="Stop Generation",
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot,
            additional_inputs=[active_system_prompt],
        )
        
        # Process prompt template changes
        system_prompt_examples.change(
            fn=update_system_prompt,
            inputs=system_prompt_examples,
            outputs=system_prompt
        )
        
        # 应用按钮更新系统提示词
        apply_button.click(
            fn=apply_system_prompt,
            inputs=[system_prompt, chatbot],
            outputs=[chatbot, chat_input, active_system_prompt, status_message],  # 添加状态消息输出
            queue=False
        )

    demo.queue(api_open=False)
    # Modify launch parameters to specify port and hostname
    try:
        demo.launch(show_api=False, share=True, server_name=args.server_name, server_port=args.server_port)
    except Exception as e:
        print(f"Server startup failed: {e}")
        print("Trying backup configuration...")
        demo.launch(show_api=False, share=False)  # Try with default configuration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/bn/lwj/IQA-project/models/Qwen2-VL-ViDA")
    parser.add_argument("--model-base", type=str, default="/mnt/bn/lwj/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    try:
        main(args)
    except ModuleNotFoundError as e:
        print(f"Module import error: {e}")
        print("It might be a Python path issue, try switching working directory...")
        
        # Try running from project root directory
        os.chdir(project_dir)
        print(f"Switch working directory to: {os.getcwd()}")
        
        # Re-import necessary modules
        print("Re-import modules...")
        from train.src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
        
        # Retry running main function
        print("Retry running main program...")
        main(args)
    except Exception as e:
        print(f"Runtime error: {e}")
        print("Please check environment configuration and retry.")
        import traceback
        traceback.print_exc()
