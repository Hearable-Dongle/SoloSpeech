<!-- <p align="center">
  <img src="assets/solospeech.jpg" width="800">
</p> -->
<img src="assets/solospeech.jpg">
<h3  align="center">🎸 SoloSpeech: Enhancing Intelligibility and Quality in Target Speech Extraction through a Cascaded Generative Pipeline</h3>

<p align="center">
 <a href="https://arxiv.org/abs/2505.19314">📜 Paper</a> | <a href="https://wanghelin1997.github.io/SoloSpeech-Demo/">🎶 Samples</a> | <a href="https://huggingface.co/spaces/OpenSound/SoloSpeech/">🤖 Spaces</a> | <a href="https://colab.research.google.com/drive/1cEcyp2rFP2DOLY4BLjaKksF48-xXQAdQ?usp=sharing">♠️ Demo</a> | <a href="https://huggingface.co/OpenSound/SoloSpeech-models/">🚀 Models</a>
</p>
<p align="center">
  <img src="https://img.shields.io/github/stars/WangHelin1997/SoloSpeech?style=social" alt="GitHub Stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg" />
</p>

## Introduction

🎯 SoloSpeech is a novel ***cascaded generative pipeline*** that integrates compression, extraction, reconstruction, and correction processes. SoloSpeech achieves state-of-the-art ***intelligibility and quality*** in target speech extraction and speech separation tasks while demonstrating exceptional ***generalization on out-of-domain data***.


[Video](https://github.com/user-attachments/assets/0b27ec4d-1a5b-446d-9ed2-43702d07b5db)

## Quick Start
- [Install and quick use](docs/quick_use.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)


## Citations

If you find this work useful, please consider contributing to this repo and cite this work:
```
@misc{wang2025solospeechenhancingintelligibilityquality,
      title={SoloSpeech: Enhancing Intelligibility and Quality in Target Speech Extraction through a Cascaded Generative Pipeline}, 
      author={Helin Wang and Jiarui Hai and Dongchao Yang and Chen Chen and Kai Li and Junyi Peng and Thomas Thebaud and Laureano Moro Velazquez and Jesus Villalba and Najim Dehak},
      year={2025},
      eprint={2505.19314},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.19314}, 
}

@inproceedings{wang2025soloaudio,
  title={SoloAudio: Target sound extraction with language-oriented audio diffusion transformer},
  author={Wang, Helin and Hai, Jiarui and Lu, Yen-Ju and Thakkar, Karan and Elhilali, Mounya and Dehak, Najim},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## License
All listening samples, source code, pretrained checkpoints, and the evaluation toolkit are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

This implementation is based on [SoloAudio](https://github.com/WangHelin1997/SoloAudio), [EzAudio](https://github.com/haidog-yaqub/EzAudio), [DPM-TSE](https://github.com/haidog-yaqub/DPMTSE), and [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). We appreciate their awesome work.

## 🌟 Like This Project?
If you find this repo helpful or interesting, consider dropping a ⭐ — it really helps and means a lot!
