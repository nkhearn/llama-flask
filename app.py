import os
import base64
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
from llama_cpp import Llama

app = Flask(__name__)

GGUF_DIR = os.path.expanduser('~/gguf')
PROMPT_DIR = os.path.expanduser('~/prompts')

llm = None

def get_llm(model_name, num_ctx, vision_enabled, n_gpu_layers):
    global llm
    model_path = os.path.join(GGUF_DIR, model_name)

    # Check if we need to reload the model
    if llm is None or llm.model_path != model_path or llm.params.n_gpu_layers != int(n_gpu_layers):
        chat_format = "llava-1-5-chatml" if vision_enabled else "chatml"
        llm = Llama(
            model_path=model_path,
            n_ctx=int(num_ctx),
            chat_format=chat_format,
            n_gpu_layers=int(n_gpu_layers),
            verbose=True,
        )
    return llm

@app.route('/')
def setup():
    models = []
    if os.path.isdir(GGUF_DIR):
        models = [f for f in os.listdir(GGUF_DIR) if f.endswith('.gguf')]
    prompts = []
    if os.path.isdir(PROMPT_DIR):
        prompts = [f for f in os.listdir(PROMPT_DIR) if f.endswith('.prompt')]
    return render_template('setup.html', models=models, prompts=prompts)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        model = request.form.get('model')
        prompt = request.form.get('prompt')
        num_ctx = request.form.get('num_ctx')
        temperature = request.form.get('temperature')
        top_k = request.form.get('top_k')
        n_gpu_layers = request.form.get('n_gpu_layers')
        vision = request.form.get('vision') == 'true'
        return render_template('chat.html', model=model, prompt=prompt, num_ctx=num_ctx, temperature=temperature, top_k=top_k, n_gpu_layers=n_gpu_layers, vision=vision)
    return redirect(url_for('setup'))

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.form
    model_name = data.get('model')
    prompt_name = data.get('prompt')
    num_ctx = data.get('num_ctx', 2048)
    temperature = float(data.get('temperature', 0.8))
    top_k = int(data.get('top_k', 40))
    n_gpu_layers = data.get('n_gpu_layers', 0)
    vision_enabled = data.get('vision') == 'true'
    user_input = data.get('user_input')
    history_str = data.get('history', '[]')
    history = json.loads(history_str)
    image_file = request.files.get('image')

    try:
        llm_instance = get_llm(model_name, num_ctx, vision_enabled, n_gpu_layers)

        system_prompt = ""
        if prompt_name:
            prompt_path = os.path.join(PROMPT_DIR, prompt_name)
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as f:
                    system_prompt = f.read()

        # Prepare messages for the LLM, ensuring the system prompt is always first
        messages_for_llm = [{"role": "system", "content": system_prompt}]

        # Add the rest of the history, filtering out any old system messages
        messages_for_llm.extend([msg for msg in history if msg.get('role') != 'system'])

        user_content = []
        if user_input:
            user_content.append({"type": "text", "text": user_input})

        if image_file and vision_enabled:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_data}"
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        if user_content:
            messages_for_llm.append({"role": "user", "content": user_content})

        response = llm_instance.create_chat_completion(messages=messages_for_llm, temperature=temperature, top_k=top_k)

        bot_response = response['choices'][0]['message']

        # Update the history for the client
        updated_history = messages_for_llm[1:] # Exclude system prompt for client-side history
        if user_content:
            updated_history[-1] = {"role": "user", "content": user_content} # Ensure the last user message is in the history
        updated_history.append(bot_response)

        return jsonify({'history': updated_history})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
