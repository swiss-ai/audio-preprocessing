SHELL := /bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH


audioprocessing:
	mv .venv-audioprocessing .venv-audioprocessing-old || true
	rm -rf .venv-audioprocessing-old &

	uv venv .venv-audioprocessing --system-site-packages

	echo 'export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$$LD_LIBRARY_PATH' >> .venv-audioprocessing/bin/activate

	uv pip compile requirements.txt -o requirements-subdeps.txt
	sed -i '/^torch==/d' requirements-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-subdeps.txt
	sed -i '/^nvidia-/d' requirements-subdeps.txt
	sed -i '/^transformer-engine/d' requirements-subdeps.txt

	source .venv-audioprocessing/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.9 && \
	uv pip install --no-deps -r requirements-subdeps.txt 
	
	