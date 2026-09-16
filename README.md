# aiml339_coVIT
Injecting locality into ViT model to improve accuracy on histopathology images


intstalling: Prerequisites (run once in your terminal, ideally inside a venv):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install transformers datasets scipy numpy


To run the program after installing the neccessary dependencies, run this in the terminal: python run_sweep_local.py

OR if python doesn't work then run py run_sweep_local.py instead, ideally in a venv.