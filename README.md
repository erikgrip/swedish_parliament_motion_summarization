# swedish_parliament_motion_summarization

Start tensorboard:
```bash
tensorboard --logdir training/logs/lightning_logs
```

Push checkpoint to Huggingface Hub:
```bash
PYTHONPATH=. python text_summarizer/save_checkpoint_to_huggingface.py --version=2 --hf_model="erikgrip2/mt5-finetuned-for-motion-title"  --hf_user="erikgrip2"
```


Run app:
```bash
PYTHONPATH=. python api_server/app.py
```

