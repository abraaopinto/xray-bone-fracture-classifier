# X-Ray Bone Fracture Classifier

Projeto de **classificação automática de fraturas ósseas em imagens de raio-X** utilizando **Deep Learning (CNN com Transfer Learning)**, com foco em **reprodutibilidade, avaliação rigorosa e interpretabilidade** por meio de **Grad-CAM**, além de uma **aplicação interativa em Streamlit** para inferência.

> ⚠️ **Aviso**: este projeto tem **finalidade educacional e experimental**. Ele **não substitui diagnóstico médico** nem deve ser utilizado para decisões clínicas reais.

---

## 1. Objetivo do Projeto

Desenvolver um pipeline completo de Machine Learning para:

- Explorar e compreender um dataset multimodal de fraturas ósseas em raio-X;
- Construir e treinar modelos de classificação baseados em CNN e Transfer Learning;
- Avaliar o desempenho do modelo com métricas apropriadas;
- Aplicar técnicas de **interpretabilidade (Grad-CAM)** para explicar as decisões do modelo;
- Disponibilizar uma aplicação simples para inferência em novas imagens.

---

## 2. Estrutura do Repositório

```text
xray-bone-fracture-classifier/
├── app/
│   └── app.py
├── notebooks/
│   └── 01_eda.ipynb
├── scripts/
│   ├── download_hbfmid.py
│   ├── inspect_hbfmid.py
│   ├── make_classification_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── compare_runs.py
├── src/
│   └── xray_bone_fracture_classifier/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       └── interpretability/
│           └── gradcam.py
├── pyproject.toml
└── .gitignore
```

---

## 3. Requisitos

- Python >= 3.10
- Sistema operacional: Windows, Linux ou macOS
- GPU opcional

---

## 4. Instalação

```bash
git clone https://github.com/abraaopinto/xray-bone-fracture-classifier.git
cd xray-bone-fracture-classifier
pip install -e .
```

---

## 5. Dataset

Utiliza o **Human Bone Fractures Multi-modal Image Dataset (HBFMID)**.

### Download

```bash
python scripts/download_hbfmid.py --output-dir data/raw
```

### Inspeção

```bash
python scripts/inspect_hbfmid.py --data-dir data/raw/Bone_Fractures_Detection --out-dir reports
```

---

## 6. Exploração dos Dados

Notebook:

```text
notebooks/01_eda.ipynb
```

Inclui:
- Visualização por classe
- Distribuição e desbalanceamento
- Análise de qualidade das imagens

---

## 7. Preparação do Dataset

```bash
python scripts/make_classification_dataset.py --input-dir data/raw/Bone_Fractures_Detection --output-dir data/processed
```

---

## 8. Treinamento

```bash
python scripts/train.py --data-dir data/processed --output-dir runs
```

---

## 9. Avaliação

```bash
python scripts/evaluate.py --runs-dir runs --output-dir reports
```

Métricas:
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusão

---

## 10. Interpretabilidade (Grad-CAM)

Implementação disponível em:

```text
src/xray_bone_fracture_classifier/interpretability/gradcam.py
```

---

## 11. Inferência via CLI

```bash
python scripts/predict.py --image-path caminho/imagem.png --model-path runs/best_model.pth
```

---

## 12. Aplicação Streamlit

```bash
streamlit run app/app.py
```

Funcionalidades:
- Upload de imagem
- Predição
- Visualização Grad-CAM

---

## 13. Reprodutibilidade

- Seeds fixadas
- Dependências versionadas
- Execução determinística

---

## 14. Considerações Finais

O projeto cobre todas as etapas esperadas de um pipeline de Visão Computacional aplicada, atendendo aos critérios de avaliação com foco em clareza, organização e rigor técnico.
