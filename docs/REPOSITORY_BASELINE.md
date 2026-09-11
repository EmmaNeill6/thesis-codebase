# Repository baseline - example change

This file records the verified state immediately before the project was placed
under a single benchmark repository on 2026-09-11. It is intended to distinguish
pre-existing behaviour from later migration regressions.

## Repository decision

- The benchmark code is versioned together because every experiment shares
  `dataset_specs.py` and the evaluator contract.
- The existing `TestingPart1` and `TestingPart2` paths remain unchanged in the
  baseline commit.
- The thesis, datasets, downloaded weights, environments, and raw model outputs
  are excluded from this repository.
- eWaSR is vendored source. Its upstream history and project changes were
  preserved before removing the nested repository metadata.

## eWaSR provenance

- Upstream: `https://github.com/tersekmatija/eWaSR.git`
- Upstream commit: `443bd4a35dafcebb76892e2047d19d743ccb8bce`
- Project baseline commit: `9469dcfcc6dd2a9f705ea8db4d2a6f744e9628d9`
- Local recovery bundle: `.migration-safety/ewasr-full-history-2026-09-11.bundle`
  (deliberately excluded from Git)

## Dataset-selection invariants

The digest is the SHA-256 of the ordered, newline-separated evaluation keys.

| Dataset key | Images | Paired masks | Ordered-key SHA-256 |
|---|---:|---:|---|
| `tampere` | 600 | 600 | `eb7caec3f8c360778004869a5272f430adc1c5cc3b4eaaccfa6b128b092d0c33` |
| `waternet` | 2,390 | 2,390 | `1041eb961115aa5860675e2540d19503178b3ea90138c6c721020cbe65157b27` |
| `mastr1325` | 1,325 | 1,325 | `5bf90c240af19c90400568ddee79a376b59a790605ae1ce730b75a2ed37e80bd` |
| `pohang` | 8 | 0 | `a2f261cb172d0c812f9da0d742a91b66805acd44092122cc3de5adeac8706123` |

## Existing SAM3 restart state

Dry runs using `.venv-sam3-mlx` reported:

| Dataset | Recognised complete | Total |
|---|---:|---:|
| MaSTR1325 | 1,325 | 1,325 |
| WaterNet | 2,390 | 2,390 |
| Pohang | 8 | 8 |
| Tampere-WaterSeg | 0 | 600 |

The Tampere mismatch existed before repository migration. Its stored run used a
different literal YAML hash even though its scientific settings are unchanged.

## Representative output digests

- eWaSR MaSTR1325 `0001m.png`:
  `500832258ca1a5009fec6b96fc9143367448ce33a2af30815b944f1fda835d1d`
- eWaSR Tampere `channel_mask_001.png`:
  `041633529b9890c717c73808236ba1400e4430b941ec2979e4f9afcfc8a58d6a`

## Hardware boundary

- SAM3 uses MLX/MLX-VLM on Apple Silicon.
- YOLO-World uses PyTorch MPS on Apple Silicon.
- eWaSR's current predictor chooses CUDA when available and otherwise CPU; it
  does not currently select MPS.
- Hardware backends and dependency versions are not changed as part of the Git
  migration.
