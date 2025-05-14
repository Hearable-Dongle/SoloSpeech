# SoloSpeech

Official Pytorch implementation of the paper: SoloSpeech: Enhancing Intelligibility and Quality in Target Speaker Extraction through a Cascaded Generative Pipeline.


<!-- ## TODO
- [ ] Release model weights
- [x] Release training code
- [x] Release inference code
- [ ] HuggingFace Spaces demo
- [ ] arxiv paper -->


## Environment setup
```bash
conda env create -f env.yml
conda activate solospeech
```

## Audio Compressor Training

To train a T-F Audio VAE model, please:

1. Change data path in `stable_audio_vae/configs/vae_data.txt` (any folder contains audio files).

2. Change model config in `stable_audio_vae/configs/stftvae_16k_320x.config`. 

We provide config for training audio files of 16k sampling rate,  please change the settings when you want other sampling rates.

3. Change batch size and training settings in `stable_audio_vae/defaults.ini`.

4. Run:

```bash
cd stable_audio_vae/
bash train_bash.sh
``` 

## Target Extractor Training

To train a targer extractor, please:

1. Prepare audio files following [SpeakerBeam](https://github.com/BUTSpeechFIT/speakerbeam).

2. Prepare latent features:

```bash
python dataset/extract_vae.py
```

3. Training:
```bash
accelerate launch scripts/solospeech/train-tse.py
```

## Corrector Training

To train a corrector, please run:
```bash
CUDA_VISIBLE_DEVICES=0 python corrector/train-fastgeco.py --gpus 1 --batch_size 16
```

## Test

To test TSE, please run:

```bash
python scripts/solospeech/test.py
```


## Evaluate

To calculate the metrics used in the paper, please run:

```bash
cd metircs/
python main.py
```


## License

All datasets, listening samples, source code, pretrained checkpoints, and the evaluation toolkit are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
See the [LICENSE](./LICENSE) file for details.

