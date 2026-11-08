# Q-DBSCAN vs DBSCAN classique sur APTOS 2019

Évaluation rigoureuse d'un **DBSCAN à noyau quantique (Q-DBSCAN)** face à un DBSCAN classique pour la détection de la rétinopathie diabétique sur les images de fond d'œil **APTOS 2019**, avec des baselines **k-NN** et **SVM** (classiques et quantiques) sur exactement les mêmes features.

> Question centrale : remplacer la distance euclidienne par une distance dérivée d'un noyau quantique (`ZZFeatureMap`) améliore-t-il le clustering par densité, et la différence est-elle statistiquement crédible ?

Tout le travail tient dans un seul notebook : [`valuation-de-q-dbscan-et-des-noyaux-quantiques-su.ipynb`](valuation-de-q-dbscan-et-des-noyaux-quantiques-su.ipynb).

---

## Principe

```
Image de fond d'œil
   └─► pré-traitement (recadrage, resize 224, CLAHE)
        └─► ResNet50 (features 2048-d, cache disque)
             └─► réduction (PCA ou LDA)  ──►  4 ou 8 composantes = 4 ou 8 qubits
                  └─► MinMaxScaler vers [0, a_max]  (angles)
                       └─► ZZFeatureMap (simulation exacte numpy)
                            └─► fidélité F = |⟨ψ(x)|ψ(x')⟩|²   →   distance D = √(1 − F)
                                 └─► DBSCAN / k-NN / SVM sur D (ou noyau F)
```

Les distances quantiques sont comparées à la **distance euclidienne** dans le même espace réduit (référence classique).

### Classifieur par densité (`DensityClassifierDBSCAN`)

1. DBSCAN (distance pré-calculée) sur le train, sans labels.
2. Chaque cluster prend le **label majoritaire** de ses points.
3. Un point de test hérite du cluster de son **core point le plus proche** s'il est à une distance ≤ `eps` ; sinon il est prédit par un **repli k-NN**.

La méthode `predict` renvoie aussi le masque `covered` (prédit par la densité) pour mesurer ce que fait réellement le DBSCAN, et non le repli k-NN.

---

## Ce que fait le notebook

| # | Élément | Détail |
|---|---------|--------|
| 1 | **Validation croisée stratifiée répétée** | 5 folds × 2 répétitions = 10 évaluations par modèle, moyenne ± écart-type |
| 2 | **Tests statistiques** | t-test apparié corrigé de **Nadeau-Bengio**, **McNemar exact**, **IC95 % bootstrap** de Δaccuracy |
| 3 | **Plusieurs échelles d'angles** | `[0, π]`, `[0, π/2]`, `[0, π/4]` pour traiter la périodicité du `ZZFeatureMap` |
| 4 | **Réductions** | PCA-4, PCA-8, LDA-4 (LDA ajustée sur les labels multiclasses après PCA à 64 d) |
| 5 | **Baselines** | k-NN, SVM-RBF classiques vs k-NN quantique et QSVM sur les mêmes features |
| 6 | **Part du repli k-NN** | `coverage`, `part_repli_kNN`, précision du DBSCAN « pur » sur les points couverts |
| 7 | **`eps` sans fuite** | choisi comme **percentile de la k-distance** sur un split interne 80/20 du train |
| 8 | **Simulation exacte vectorisée** | `ZZFeatureMap` en numpy, **vérifié contre Qiskit** (écart max ≈ 1e-15) |
| 9 | **Diagnostics** | monotonie (Spearman), graphes de k-distance |

Deux tâches sont évaluées : **binaire** (sain / malade) et **multiclasse** (5 grades de 0 à 4).

**Anti-fuite** : tout ce qui est ajusté (scaler, PCA, LDA, MinMax, `eps`, `C`) l'est uniquement sur le train du fold, et sur un sous-ensemble interne du train pour le choix des hyper-paramètres.

---

## Résultats principaux

Moyennes sur 10 folds, tâche **binaire** (accuracy ; F1-macro quasi identique). Q-DBSCAN et QSVM : meilleure échelle d'angles parmi celles testées.

| Config | DBSCAN classique | Q-DBSCAN | k-NN classique | k-NN quantique | SVM-RBF | QSVM |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| PCA-4 | 0.854 | 0.841 | 0.855 | 0.844 | 0.866 | 0.866 |
| PCA-8 | 0.914 | 0.858 | 0.927 | 0.881 | 0.934 | 0.929 |
| LDA-4 | 0.925 | 0.927 | 0.929 | 0.927 | 0.932 | 0.931 |

Lecture :

- **PCA-4 et PCA-8** : le Q-DBSCAN est significativement **moins bon** que le DBSCAN classique (IC95 % de Δacc strictement négatif, McNemar significatif). Le QSVM, lui, rejoint le SVM-RBF pour les petites plages d'angles.
- **LDA-4** : les écarts quantique / classique deviennent **statistiquement indiscernables** (les IC95 % contiennent 0) ; on gagne en revanche en `coverage` du DBSCAN.
- **Aucun avantage quantique** n'est observé sur ces données : au mieux la parité.
- Le DBSCAN est peu discriminant : en PCA, ~**80 %** des prédictions passent par le repli k-NN (`coverage` ≈ 0.20). Avec LDA-4, la part du repli descend à 62–75 %. La performance reflète donc en grande partie celle d'un k-NN.

### Diagnostic de monotonie (PCA-4, un fold)

Corrélation de Spearman entre la distance euclidienne dans l'espace des angles et la distance quantique :

| Plage d'angles | ρ |
|---|:---:|
| `[0, π]` | 0.396 |
| `[0, π/2]` | 0.674 |
| `[0, π/4]` | 0.824 |

Plus la plage est large, plus la périodicité du `ZZFeatureMap` rend la distance quantique **non monotone** : des points éloignés peuvent paraître proches, ce qui nuit à DBSCAN.

> Les chiffres ci-dessus viennent des sorties enregistrées dans le notebook (graine 42, 2930 images, `qiskit` installé via pip). Ils peuvent varier légèrement selon les versions de bibliothèques.

---

## Données

[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) : 2930 images de fond d'œil annotées en 5 grades.

| Grade | 0 | 1 | 2 | 3 | 4 |
|-------|:-:|:-:|:-:|:-:|:-:|
| Images | 1434 | 300 | 808 | 154 | 234 |

- **Binaire** : grade 0 = sain, grades 1 à 4 = malade.
- Le notebook attend un `train_1.csv` (colonnes `id_code`, `diagnosis`) et le dossier `train_images/` : les chemins par défaut sont ceux d'un notebook **Kaggle** (`/kaggle/input/...`).

Les données ne sont **pas** incluses dans ce dépôt.

---

## Installation et exécution

### Sur Kaggle (recommandé)

1. Créer un notebook Kaggle et importer le fichier `.ipynb`.
2. Ajouter le dataset APTOS 2019 en *Input* (le notebook cherche `train_1.csv` et `train_images`).
3. Activer un GPU pour l'extraction ResNet50 (facultatif, ~CPU possible).
4. Tout exécuter. La validation croisée complète (3 configurations × 10 folds × 4 espaces de distance) a pris environ **6,7 min** dans la version enregistrée.

### En local

```bash
pip install qiskit numpy pandas scipy scikit-learn matplotlib seaborn tqdm \
            torch torchvision opencv-python
```

Puis adapter les deux chemins de la section 2 (`CSV_PATH`, `IMG_DIR`) et `CACHE_PATH` de la section 3 (par défaut `/kaggle/working/...`).

Les features ResNet50 sont mises en **cache** (`.npz`, indexé par les `id_code`) pour ne pas être recalculées.

---

## Configuration

Tous les paramètres sont regroupés dans la cellule de configuration (section 1) :

| Paramètre | Valeur par défaut | Rôle |
|-----------|-------------------|------|
| `SEED` | `42` | Reproductibilité |
| `N_SAMPLES` | `2930` | Nombre d'images (`None` = toutes) |
| `N_SPLITS` / `N_REPEATS` | `5` / `2` | Validation croisée répétée |
| `EXPERIMENTS` | `[("pca",4), ("pca",8), ("lda",4)]` | (réduction, nombre de qubits) |
| `ANGLE_MAXES` | `π`, `π/2`, `π/4` | Plage des angles après `MinMaxScaler` |
| `REPS` | `1` | Répétitions du `ZZFeatureMap` |
| `MIN_SAMPLES` | `5` | `MinPts` de DBSCAN (point lui-même compris) |
| `KNN_FALLBACK` | `5` | `k` du repli k-NN |
| `Q_GRID` | `5, 10, …, 95` | Percentiles de k-distance testés pour `eps` |
| `SVM_C_GRID` | `[0.1, 1, 10, 100]` | Grille de `C` pour SVM / QSVM |
| `NOISE_MAX` | `0.80` | Rejette un `eps` qui met plus de 80 % des points en bruit |

---

## Structure du notebook

| Section | Contenu |
|---------|---------|
| 0 – 1 | Installation, imports, configuration |
| 2 | Chargement APTOS 2019 (sous-échantillon stratifié) |
| 3 | Pré-traitement et extraction ResNet50 (avec cache) |
| 4 | Simulation exacte du `ZZFeatureMap` + vérification contre Qiskit |
| 5 | Espaces de distance (réduction → angles → noyau quantique) |
| 6 | Classifieurs (`DensityClassifierDBSCAN`, k-NN, SVM / QSVM) |
| 7 | Expériences (validation croisée répétée) |
| 8 | Résultats : moyenne ± écart-type, diagnostic du DBSCAN |
| 9 | Tests statistiques (Nadeau-Bengio, McNemar, bootstrap) |
| 10 | Diagnostics (monotonie, k-distance) |
| 11 – 12 | Vue d'ensemble et guide de lecture |

---

## Limites

- Le noyau quantique est **simulé exactement** (pas de bruit, pas de shots) : on mesure l'effet de la *métrique*, pas celui d'un vrai processeur.
- Les features ResNet50 sont **figées** (pas de fine-tuning). Un fine-tuning devrait être refait dans chaque fold pour éviter toute fuite, ce qui multiplie le coût de calcul.
- Les performances DBSCAN dépendent fortement du repli k-NN : à lire avec `coverage` et `acc_covered`, pas seulement l'accuracy globale.
- Avec 2930 images et 10 évaluations par modèle, les petits écarts (< 1 point) ne sont pas interprétables sans les tests de la section 9.

---

## Références

- Havlíček *et al.*, *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 2019.
- Ester *et al.*, *A density-based algorithm for discovering clusters in large spatial databases with noise* (DBSCAN), KDD 1996.
- Nadeau & Bengio, *Inference for the generalization error*, Machine Learning 52, 2003.
- Xuming Xie, Longzhen Duaa, Taorong Qiu & Junru Li, *Quantum algorithm for MMNG‑based DBSCAN*
- Kaggle, *APTOS 2019 Blindness Detection*.

## Licence

À définir par l'auteur du dépôt. Le jeu de données APTOS 2019 reste soumis aux conditions de la compétition Kaggle.
