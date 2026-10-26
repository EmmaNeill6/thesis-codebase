# Segmentation evaluation

The two evaluation tasks have separate code and result folders:

```text
evaluation/
├── binary/
│   ├── water_mask_evaluator.py
│   ├── water_metrics.py
│   ├── compare_model_performance.py
│   ├── test_compare_model_performance.py
│   └── results/                    # Generated locally; ignored
├── three_class/
│   ├── three_class_evaluator.py
│   ├── class_metrics.py
│   ├── test_three_class_evaluator.py
│   └── results/                    # Created when three-class evaluation runs
└── requirements.txt                # Shared evaluation dependencies
```

Use [binary evaluation](binary/README.md) for water versus non-water masks.
Use [three-class evaluation](three_class/README.md) for obstacle, water and sky.
Neither evaluator runs a model; both evaluate saved prediction masks.

Run examples from the `thesis-codebase` directory with a Python environment
containing the dependencies in `evaluation/requirements.txt`. The existing
`.venv-sam3-mlx/bin/python` environment contains these dependencies.

Generated evaluation runs are ignored. Selected compact historical evidence is
published under `results/` at the repository root. The established evaluators
exclude the later uncommitted boundary-metric additions.

```sh
python -m unittest discover -s evaluation -v
```
