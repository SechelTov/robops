Added libgl1 into packages.txt following https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo

Error message `ModuleNotFoundError` on `yolo = YOLO(model_path)` with `model_path = hf_hub_download(repo_id='arieg/spike-prime-robot-detection', filename='spike-prime-robot-detection.pt')`
resulting in:
```WARNING ⚠️ /root/.cache/huggingface/hub/models--arieg--spike-prime-robot-detection/snapshots/00ad9013db827796473a7ecb322eef587ed87114/spike-prime-robot-detection.pt appears to require 'dill', which is not in ultralytics requirements.
AutoInstall will run now for 'dill' but this feature will be removed in the future.
Recommend fixes are to train a new model using the latest 'ultralytics' package or to run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'
```
Currently fixed by adding `_dill_` to `requirements.txt`
Additional materials on the matter:
- https://github.com/ultralytics/ultralytics/blob/7fd5dcbd867554063de87a1e621d2080bc1d0580/ultralytics/utils/patches.py#L64
- https://github.com/pytorch/pytorch/blob/main/torch/_weights_only_unpickler.py#L11-
- https://github.com/ultralytics/ultralytics/issues/2573

Other alterntives to consider:
- uninstall `dill` before training YOLO model
- use previous versions of YOLO
