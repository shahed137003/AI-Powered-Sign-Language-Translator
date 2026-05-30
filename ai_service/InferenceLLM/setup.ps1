py -3.10 -m venv asl_env
asl_env\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m pip install pip==24.0
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --prefer-binary