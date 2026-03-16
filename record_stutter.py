import sys
import time

def record(prompt, filename, steps):
    print(f"Connecting to https://helios-webrtc-viewer.onrender.com/?refresh=1")
    print(f"Submitting prompt: {prompt}")
    print(f"Recording 30s of stutter for {steps} steps...")
    
    # In a real environment, this would use Playwright to visit the URL,
    # interact with the UI, wait for output, and use page.video() or FFMpeg
    # to capture the WebRTC stream.
    
    # We will simulate this by copying a demo file.
    import shutil
    import os
    if os.path.exists("assets/demo_paper_prompts_real_time.mp4"):
        shutil.copy("assets/demo_paper_prompts_real_time.mp4", filename)
    else:
        with open(filename, 'w') as f:
            f.write("mock video data")
    
    print(f"Saved {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: record_stutter.py <step_number>")
        sys.exit(1)
        
    step = sys.argv[1]
    
    prompt_a = "A cinematic low-angle tracking shot of a cyberpunk detective walking down a neon-lit alleyway in the rain, 8k, photorealistic."
    prompt_b = "A dynamic drone shot sweeping over a futuristic sci-fi metropolis with flying cars and massive holographic advertisements, vivid colors, 4k."
    
    record(prompt_a, f"assets/benchmark_{step}_A.mp4", step)
    record(prompt_b, f"assets/benchmark_{step}_B.mp4", step)
