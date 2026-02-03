# X-Ray Bone Fracture Classifier

Pipeline completo em **Python + PyTorch** para **classificação de fraturas ósseas em radiografias**, incluindo:
- preparação de dados,
- treino e avaliação,
- inferência single-image,
- explicabilidade via **Grad-CAM**,
- aplicação interativa em **Streamlit**,
- testes automatizados e boas práticas de engenharia.

---

## 1. Visão Geral

Este projeto implementa um **classificador supervisionado de fraturas ósseas** a partir de imagens de raio-X.
O foco não é apenas o modelo, mas **todo o ciclo de vida**:

- **Reprodutibilidade** (configs, histórico, artefatos)
- **Robustez** (validação, erros amigáveis)
- **Explicabilidade** (Grad-CAM)
- **Qualidade de código** (lint, testes, CI-ready)

---

## 2. Estrutura do Projeto

```text
.
├── app/                       # Streamlit app
│   └── app.py
├── data/
│   ├── raw/                   # dados brutos (download)
│   └── processed/             # dataset preparado para treino
├── models/
│   └── run_YYYYMMDD-HHMMSS/    # artefatos de cada treino
│       ├── model.pt
│       ├── config.json
│       ├── labels.json
│       ├── train_history.csv
│       └── train_summary.json
├── reports/
│   └── app_runs/              # saídas do app (predições)
├── scripts/                   # CLIs do pipeline
│   ├── download_hbfmid.py
│   ├── make_classification_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── src/
│   └── xray_bone_fracture_classifier/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── inference/
│       ├── evaluation/
│       ├── interpretability/
│       └── utils/
├── tests/                     # testes automatizados
├── pyproject.toml
└── README.md
```

---

## 3. Instalação

### 3.1 Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3.2 Instalar dependências

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 4. Pipeline End-to-End (Copy & Paste)

### 4.1 Download do dataset (exemplo)

```bash
python scripts/download_hbfmid.py --out-dir data/raw
```

> Ajuste conforme a origem real do dataset.

---

### 4.2 Preparar dataset de classificação

```bash
python scripts/make_classification_dataset.py \
  --data-dir data/raw \
  --out-dir data/processed/hbfmid_cls_bbox \
  --img-size 224
```

---

### 4.3 Treino (smoke test – 1 época)

```bash
python scripts/train.py \
  --data-dir data/processed/hbfmid_cls_bbox \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0
```

---

### 4.4 Avaliação

```bash
python scripts/evaluate.py \
  --model-dir models/run_YYYYMMDD-HHMMSS \
  --data-dir data/processed/hbfmid_cls_bbox
```

---

### 4.5 Predição single-image (CLI)

```bash
python scripts/predict.py \
  --model-dir models/run_YYYYMMDD-HHMMSS \
  --image path/para/xray.png \
  --topk 3
```

---

## 5. Aplicação Interativa (Streamlit)

```bash
streamlit run app/app.py
```

---

## 6. Testes e Qualidade

```bash
pytest -q
ruff check .
```

---

## Licença

Uso educacional / demonstrativo.
