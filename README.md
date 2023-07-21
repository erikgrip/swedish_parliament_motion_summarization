# swedish_parliament_motion_summarization

Start tensorboard:
```bash
tensorboard --logdir training/logs/lightning_logs
```

Push checkpoint to Huggingface Hub:
```bash
PYTHONPATH=. python motion_title_generator/save_checkpoint_to_huggingface.py --version=2 --hf_model="erikgrip2/mt5-finetuned-for-motion-title"  --hf_user="erikgrip2"
```

Build app Docker image:
```bash
# Navigate to project's root directory and run
docker build -t motion_title_app -f api_server/Dockerfile .
```

Run app in Docker container:
```bash
docker run -p 8000:8000 motion_title_app
```

Run app without docker:
```bash
PYTHONPATH=. python api_server/app.py
```

Make build script executable if needed:
```bash
chmod +x api_server/build_app_image.sh
```



