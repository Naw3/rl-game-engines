# Ultimate AlphaZero Overhaul Plan (Transformer Edition)

Puisque tu es prêt à casser l'existant pour viser l'excellence absolue, voici le plan ultime basé sur les toutes dernières recherches en IA (2023-2024), en particulier la transition réussie de *Leela Chess Zero* vers les architectures **Transformer**.

## User Review Required

> [!CAUTION]
> L'implémentation de ce plan va totalement remplacer le réseau de neurones actuel (`ResNet`) par un réseau à base d'attention spatiale (`Transformer / Conformer`). L'ancien checkpoint `connect4_model.pt` sera inutilisable, et l'apprentissage repartira de ZÉRO. 

## 1. Nouvelle Architecture : Spatial Board Transformer
Les réseaux convolutionnels (CNN / ResNet) sont très lents pour comprendre que le jeton en bas à gauche bloque un piège en haut à droite (besoin de nombreuses couches pour faire voyager l'information).
**La solution :** Un modèle basé sur **l'Attention Linéaire (Conformer / ViT)**.
- Le plateau (6x7 = 42 cases) sera découpé en **42 tokens**.
- Le mécanisme de *Self-Attention* permettra au réseau de voir instantanément les relations globales entre toutes les cases du plateau en une seule couche.
- Le modèle aura **3 têtes de sortie** :
  1. `Policy` : La probabilité des 7 coups.
  2. `Value` : La probabilité de victoire (win/loss).
  3. `Moves Left` (Auxiliary Loss) : Le nombre de coups avant la fin de la partie.

## 2. Optimisations MCTS "State-of-the-Art" (Rust)
Ces techniques vont propulser la vitesse de génération et la qualité de l'arbre de recherche.

### A. Playout Cap Randomization (PCR)
- 75% des coups joués dans l'auto-play utiliseront seulement **30 simulations** (recherche ultra-rapide).
- 25% des coups utiliseront **200 simulations** (recherche profonde).
- **Résultat :** Le réseau génère 3x plus de parties par seconde, accélérant drastiquement l'apprentissage de la tête "Value".

### B. Dynamic First Play Urgency (FPU)
- Les nœuds non-visités dans l'arbre ne seront plus initialisés à `0`, mais à `Valeur_du_Parent - 0.1`.
- **Résultat :** Le moteur ne perd plus de temps à explorer des coups considérés comme perdants par le Transformer.

### C. Maximum Backpropagation (Minimax Hybrid)
- Si le MCTS trouve une victoire absolue (alignement de 4), il remontera une valeur stricte (`1.0`) plutôt que de moyenner la victoire avec les autres branches (`W/N`).
- **Résultat :** Le modèle apprend à repérer les tactiques mortelles sans hésitation.

---

## Plan d'Exécution Technique

### [MODIFY] `src_python/model.py`
- Supprimer l'architecture ResNet (`ResidualBlock`).
- Créer une nouvelle classe `Connect4Transformer` utilisant `nn.TransformerEncoder` ou un module d'attention personnalisé.
- Ajouter la perte auxiliaire (MSE Loss) pour la tête `Moves Left`.

### [MODIFY] `src_rust/src/main.rs`
- Coder la logique PCR (génération d'un nombre aléatoire pour décider du seuil de simulations).
- Modifier la formule `PUCT` pour inclure le *Dynamic FPU*.
- Modifier la propagation des valeurs (*backpropagation*) pour le *Maximum Backprop*.
- Exporter la valeur cible `moves_left` dans la trame binaire `C4D1`.

### [MODIFY] `src_python/train.py`
- Mettre à jour le pipeline d'entraînement pour lire la 3ème cible et l'optimiser.

## Verification Plan
1. Vérifier que la nouvelle architecture compile bien en ONNX et se charge dans Rust sans erreur.
2. Lancer `bench_cycle.ps1` et vérifier que le FPS MCTS a explosé grâce au PCR.
3. Vérifier que les 3 fonctions de perte (Policy, Value, Moves Left) convergent pendant l'entraînement.
