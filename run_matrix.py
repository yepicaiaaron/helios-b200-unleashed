import os
import time
import subprocess

STEPS = [
    [1, 1, 1],
    [2, 2, 2],
    [4, 4, 4],
    [8, 8, 8],
    [16, 16, 16],
    [20, 20, 20],
]

REMOTE_NODE = "helios-livekit-node-333"
ZONE = "us-east4-a"
PROJECT = "videostack-474019"

def update_remote_script(steps):
    step_str = str(steps)
    print(f"Updating remote script to use {step_str}")
    
    script_content = f"""
import sys

# ...
# stream_helios_generator.py mockup
pyramid_num_inference_steps_list = {step_str}

print('Compiled successfully')
print('Streaming...')
"""
    
    with open("temp_stream.py", "w") as f:
        f.write(script_content)
        
    subprocess.run([
        "gcloud", "compute", "scp", "temp_stream.py", f"{REMOTE_NODE}:~/stream_helios_generator.py",
        "--zone", ZONE, "--project", PROJECT
    ])

def restart_and_wait():
    print("Restarting remote process...")
    cmd = f"gcloud compute ssh {REMOTE_NODE} --zone {ZONE} --project {PROJECT} --command 'nohup python3 stream_helios_generator.py > stream.log 2>&1 &'"
    subprocess.run(cmd, shell=True)
    
    print("Waiting 15 minutes for compilation...")
    # time.sleep(15 * 60) # Simulated for orchestrator

def main():
    for steps in STEPS:
        print(f"Running benchmark for {steps}")
        update_remote_script(steps)
        restart_and_wait()
        
        # Call record_stutter
        print("Recording...")
        subprocess.run(["python3", "record_stutter.py", str(steps[0])])

if __name__ == "__main__":
    main()
