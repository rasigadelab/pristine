#%%
##############################
# RUN ALL DEMOS
# Use debugging mode (F5) to catch regressions early
##############################
import os
import subprocess
import concurrent.futures

examples_folder = "examples"
scripts = [
    os.path.join(examples_folder, f) 
    for f in os.listdir(examples_folder) 
    if f.endswith('.py') and f.find("run_all")<0]

def run_script(script):
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
    print(f"Finished {script}")

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_script, scripts)
# %%
