# 使用官方Python运行时作为父镜像
FROM huggingface/transformers-pytorch-gpu:latest

# 设置工作目录
WORKDIR /app

# 将当前目录内容复制到容器中的/app目录
COPY ./requirements.txt /app

# Set timezone
ENV TZ=Asia/Hong_Kong
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata

# 安装任何需要的包
RUN conda install nvidia/label/cuda-11.8.0::cuda

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# 使start.sh脚本可执行
RUN chmod +x /app/start.sh

# 如果需要的话，可以在这里添加其他系统依赖项的安装命令
# RUN apt-get update && apt-get install -y ...

# 暴露你的应用所需要的端口（如果适用）
EXPOSE 1216

# 使用CMD指定默认命令，或者使用ENTRYPOINT来强制容器运行该脚本
CMD ["bash"]
# ENTRYPOINT ["./start.sh"]