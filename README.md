# Classification d'images de fleurs (102 classes) par Transfer Learning

Classification de 102 espèces de fleurs avec MobileNetV2 pré-entraîné sur ImageNet, adapté par fine-tuning.

## Problématique

Reconnaître automatiquement l'espèce d'une fleur à partir d'une photo. Avec 102 classes visuellement proches, entraîner un réseau de zéro demanderait beaucoup de données et de temps de calcul : le Transfer Learning permet d'obtenir de bons résultats avec un modèle léger.

## Données

- **Source :** [The Oxford Flowers 102 Dataset (Kaggle)](https://www.kaggle.com/datasets/waseemalastal/the-oxford-flowers-102-dataset)
- **102 classes** d'espèces de fleurs (rose, tournesol, lotus, hibiscus, etc.)
- **Entraînement :** 6 552 images
- **Validation :** 818 images
- Un dossier `test` est fourni avec le dataset, mais il n'est pas utilisé dans ce notebook.

## Méthode

1. **Prétraitement :** images redimensionnées en 224 × 224, normalisation avec `preprocess_input` de MobileNetV2.
2. **Augmentation de données** (entraînement uniquement) : rotation, décalages, cisaillement, zoom, retournement horizontal.
3. **Modèle :**
   - MobileNetV2 pré-entraîné sur ImageNet (sans la couche de classification)
   - **Fine-tuning** : seules les 50 dernières couches sont entraînées, les autres sont gelées
   - `GlobalAveragePooling2D` → `Dense(128, ReLU)` → `Dense(102, softmax)`
   - 2,4 millions de paramètres, dont 2,0 millions entraînables
4. **Entraînement :** optimiseur Adam (learning rate 1e-4), perte `categorical_crossentropy`, 10 époques, batch de 32.

## Résultats

| Métrique | Entraînement | Validation |
|---|---|---|
| Exactitude (dernière époque) | 99,6 % | **94,7 %** |
| Perte (dernière époque) | 0,025 | 0,180 |

Meilleure exactitude en validation : 95,4 % (époque 9).

**Lecture :** le modèle reconnaît correctement environ 95 % des images de validation sur 102 classes. L'écart avec l'entraînement (99,6 %) indique un léger surapprentissage.

## Limites et pistes d'amélioration

- Évaluation sur le jeu de validation uniquement : évaluer sur le dossier `test` fourni donnerait une mesure finale plus fiable.
- Léger surapprentissage : tester plus d'augmentation, du dropout ou un arrêt anticipé (`EarlyStopping`).
- Pas d'analyse par classe : ajouter une matrice de confusion pour repérer les espèces confondues.
- Comparer avec d'autres architectures (EfficientNet, ResNet) et ajuster le nombre de couches dégelées.

## Structure du dépôt

```
classification-fleurs-mobilenetv2/
├── README.md
├── classification_fleurs.ipynb
├── requirements.txt
└── .gitignore
```

## Installation et utilisation

```bash
git clone https://github.com/wissalelhallaoui-data/classification-fleurs-mobilenetv2.git
cd classification-fleurs-mobilenetv2
pip install -r requirements.txt
jupyter notebook classification_fleurs.ipynb
```

Le jeu de données est téléchargé automatiquement depuis Kaggle avec `kagglehub`. Un GPU est recommandé (le notebook a été exécuté sur Google Colab).

## Technologies

Python, TensorFlow/Keras, MobileNetV2, NumPy, Matplotlib

## Auteure

**Wissal EL HALLAOUI**, étudiante en M2 Ingénierie des Systèmes Intelligents
[LinkedIn](https://www.linkedin.com/in/wissal-el-hallaoui-a8b9bb29a)
