docker run -it --mount "type=bind,source=$(pwd),target=/app/merge-rl" --entrypoint /bin/bash --gpus=all dt-merging
