import gradio as gr

def text2audio(text, duration, guidance_scale, random_seed, n_candidates, model_name):
    global audioldm, current_model_name
    
    if audioldm is None or model_name != current_model_name:
        audioldm=build_model(model_name=model_name)
        current_model_name = model_name
        
    # print(text, length, guidance_scale)
    waveform = text_to_audio(
        latent_diffusion=audioldm,
        text=text,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )  # [bs, 1, samples]
    waveform = [
        gr.make_waveform((16000, wave[0]), bg_image="bg.png") for wave in waveform
    ]
    # waveform = [(16000, np.random.randn(16000)), (16000, np.random.randn(16000))]
    if len(waveform) == 1:
        waveform = waveform[0]
    return waveform

textbox_input = gr.Textbox(
    value="A hammer is hitting a wooden surface",
    max_lines=1,
    label="Input your text here. Please ensure it is descriptive and of moderate length.",
)

duration_input = gr.Slider(
    2.5,
    10,
    value=5,
    step=2.5,
    label="Duration (seconds)"
)

guidance_scale_input = gr.Slider(
    0,
    5,
    value=2.5,
    step=0.5,
    label="Guidance scale (Large => better quality and relevance to text; Small => better diversity)"
)

random_seed_input = gr.Number(
    value=42,
    label="Change this value (any integer number) will lead to a different generation result."
)

n_candidates_input = gr.Slider(
    1,
    5,
    value=3,
    step=1,
    label="Automatic quality control. This number controls the number of candidates (e.g., generate three audios and choose the best to show you). A larger value usually leads to better quality with heavier computation"
)

model_name_input = gr.Dropdown(
    ["audioldm-s-full", "audioldm-l-full", "audioldm-s-full-v2","audioldm-m-text-ft", "audioldm-s-text-ft", "audioldm-m-full"],
    value="audioldm-m-full",
    label="Choose the model to use. audioldm-m-text-ft and audioldm-s-text-ft are recommended. -s- means small, -m- means medium and -l- means large"
)

output_video = gr.Video(label="Output", elem_id="output-video")

interface_layout = gr.Column(
    textbox_input,
    duration_input,
    guidance_scale_input,
    random_seed_input,
    n_candidates_input,
    model_name_input,
    output_video
)

iface = gr.Interface(
    fn=text2audio,
    inputs=[textbox_input, duration_input, guidance_scale_input, random_seed_input, n_candidates_input, model_name_input],
    outputs=[output_video],
    allow_flagging="never",
    title="AudioLDM: Text-to-Audio Generation with Latent Diffusion Models",
    description="Input your text and configure parameters to generate audio.",
    layout=interface_layout
)

iface.launch()
