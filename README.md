# ad-video-classifier

Training and inference code for a multimodal model that classifies advertising
videos by production format. The model predicts one of three classes: UGC,
motion graphics, and live action.

## Usage

```bash
pip install -r requirements.txt

huggingface-cli download MADUP/ad-video-classifier adapter_model.bin --local-dir .
python inference.py sample.mp4
```

Weights are published at
[MADUP/ad-video-classifier](https://huggingface.co/MADUP/ad-video-classifier).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgement

This work was supported by the Ministry of Science and ICT (MSIT) and the Korea
Association for ICT Promotion (KAIT) under the Advanced GPU Utilization Support
Program.
