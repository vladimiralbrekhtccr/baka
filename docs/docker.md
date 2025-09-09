# Docker


1. How to build docker for vLLM

```bash
sudo DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile . \
  --target vllm-openai \
  --tag vllm-oylan-2_5:0.9.2-cu122 \
  --build-arg CUDA_VERSION=12.2.0 \
  --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04 \
  --build-arg max_jobs=33 \
  --build-arg nvcc_threads=2 \
  --build-arg RUN_WHEEL_CHECK=false
```


2. Tar it

```bash
docker save -o vllm-oylan-2_5_0.9.2-cu124.tar vllm-oylan-2_5:0.9.2-cu124
```

3. Upload

```bash
huggingface-cli upload issai/docker_images_avlm_vLLM-int_ver1 vllm-oylan-2_5_0.9.2-cu124.tar
```

4. Download

```bash
wget --header="Authorization: Bearer <YOUR_HF_TOKEN>" \
  "https://huggingface.co/issai/docker_images_avlm_vLLM-int_ver1/resolve/main/vllm-oylan-2_5_0.9.2-cu124.tar" \
  -O vllm-oylan-2_5_0.9.2-cu124.tar
```

5. Load it 

```bash
docker load -i vllm-oylan-2_5_0.9.2-cu124.tar
```

6. Use it

```bash
bash docker_info/run_vllm_server_docker.sh
```

7. ybit docker step by step:

```bash
docker images
docker stop ki_oylan_a_v_t_2_5
docker rm ki_oylan_a_v_t_2_5
```