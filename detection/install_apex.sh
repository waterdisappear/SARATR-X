git clone https://github.com/NVIDIA/apex.git ../apex
cd ../apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
