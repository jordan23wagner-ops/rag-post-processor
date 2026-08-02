# Base image: https://hub.docker.com/r/apify/actor-python
FROM apify/actor-python:3.14

USER myuser

# Copy requirements first so dependency install is cached independently of
# source changes.
COPY --chown=myuser:myuser requirements.txt ./

RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies:" \
 && pip install --no-cache-dir -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze

# Bake the tiktoken BPE tables into the image (~5 MB for both encodings).
# Without this, the first tokenizer call of every run fetches them over HTTPS
# from openaipublic.blob.core.windows.net; if that host is slow or blocked the
# run degrades to estimated token counts, or fails.
ENV TIKTOKEN_CACHE_DIR=/home/myuser/.tiktoken
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
 && python -c "import tiktoken; [tiktoken.get_encoding(e) for e in ('cl100k_base','o200k_base','p50k_base','r50k_base')]" \
 && echo "Cached tiktoken encodings:" && ls -la "$TIKTOKEN_CACHE_DIR"

COPY --chown=myuser:myuser . ./

RUN python -m compileall -q rag_post_processor/

CMD ["python", "-m", "rag_post_processor"]
