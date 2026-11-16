# U-Net training result — Kazakhstan

## Run information

- Date: 2026-08-24
- Task: Satellite-image segmentation for Kazakhstan drought-risk monitoring
- Model: U-Net
- Device: CPU
- Epochs: 20
- Best epoch: 17
- Best validation loss: 0.872854

## Training configuration

- Patch size: `256 × 256` pixels; images are loaded as `C × H × W` tensors.
- Input channels/bands: `7` — Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI.
- Normalization: bands 1–6 use `clip((x + 0.2) / 1.2, 0, 1)`; NDVI uses `clip((x + 1.0) / 2.0, 0, 1)`.
- Training augmentation: horizontal flip, vertical flip, and random 90° rotation, each with probability `0.5`.
- Batch size: `4`.
- Learning rate: `1e-3`.
- Optimizer: `torch.optim.AdamW`, initialized with the learning rate above; no other optimizer arguments are passed in the code.
- Loss function: `BCEWithLogitsLoss + soft Dice loss`; Dice smoothing is `1.0`.
- Prediction-mask threshold: `sigmoid(logits) >= 0.5`.
- Drought-label threshold: `VCI < 35.0%`.
- Model architecture: `segmentation_models_pytorch.Unet` with a `resnet34` encoder, `encoder_weights=None`, `in_channels=7`, `classes=1`, and `activation=None`.
- Epochs: `20`; data-loader workers: `0`.
- Random seed: the patch split uses `random.Random(42)`; these modules do not set a PyTorch or NumPy training seed.
- Train/validation/test split: `70% / 15% / 15%`, assigned by the seeded patch-export split.

## Dataset split

- Train patches: 59
- Validation patches: 11
- Test patches: 9
- Total patches: 79

## Final test metrics

| Metric | Value |
|---|---:|
| Test loss | 0.861280 |
| Dice | 0.827955 |
| IoU | 0.706419 |
| Precision | 0.782883 |
| Recall | 0.878534 |

## Best checkpoint

- File: `ml/weights/best_model.pth`
- Selected by: lowest validation loss
- Best checkpoint validation loss: 0.872854

## Training log

- Log file: `ml/weights/pipeline-output-retry.log`

## Notes

- Training was run on CPU.
- The checkpoint from epoch 17 was selected as the best model.
- The model achieved Dice 0.827955 and IoU 0.706419 on the test split.
- Recall is higher than precision, so the model finds most target areas but can include extra pixels.

## City-dataset fine-tuning — 2026-11-16

- Dataset: `data/patches_kazakhstan_cities_2024` (45 paired patches: 30 train, 6 validation, 9 test).
- The city-based split keeps the sampled cities disjoint: 10 train cities, 2 validation cities, and 3 test cities.
- Initialization: the previous deployed seven-channel ResNet34 U-Net checkpoint (epoch 31).
- Fine-tuning: 20 epochs, batch size 4, learning rate `1e-4`, CPU.
- Best checkpoint: fine-tuning epoch 2 (cumulative checkpoint epoch 33), selected by validation loss.
- Validation loss on the new validation split changed from `0.687350` before fine-tuning to `0.685597`.
- Test metrics after fine-tuning: loss `1.102817`, Dice `0.655405`, IoU `0.487437`, precision `0.576260`, recall `0.759752`.
- Baseline Dice on that same test split was `0.660443`; the test Dice was slightly lower after fine-tuning. The new checkpoint was selected for its lower validation loss, not for a test-set improvement.
- The previous deployed checkpoint is preserved at `ml/weights/best_model_before_city_finetune_20261116.pth`.
- Console and epoch log: `ml/weights/city-finetune-20261116.log`.

These are small, geographically sampled datasets and VCI-derived proxy labels, not field-verified drought observations. Treat the results as an educational prototype, not a validated operational product.
