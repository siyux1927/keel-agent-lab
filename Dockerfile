# 单阶段就够: 全部依赖是纯 Python(离线 Mock 不需要任何模型权重或原生扩展),
# 镜像小到没必要为了几十 MB 去做多阶段构建。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先只拷依赖清单: 源码改动不会让 pip 层失效, 重建镜像快得多
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY keel/ ./keel/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./
RUN mkdir -p data

# 非 root 运行。这个服务里有 python_exec 和 read_file 两个工具, 虽然都做了
# AST 白名单和路径校验, 但纵深防御的第一层永远是"进程本身权限就不够"。
RUN useradd --create-home --uid 10001 keel && chown -R keel:keel /app
USER keel

EXPOSE 8000

# 公网部署默认打开自我保护(限流 + 输入长度上限), 本地跑容器可以用
# -e KEEL_PUBLIC_MODE=false 关掉
ENV KEEL_PUBLIC_MODE=true \
    KEEL_PROVIDER=mock

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "keel.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
