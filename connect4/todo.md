# TODO — INT8 / TensorRT pour Connect4

## Contexte

Le benchmark GPU montre que `MatMulInteger` via Rust + ONNX Runtime + CUDA
peut atteindre environ `5,5x` contre FP32 sur une grosse matrice 4096×4096.

En revanche, le modèle Connect4 complet quantifié dynamiquement avec
`ConvInteger` ne fonctionne pas correctement en self-play GPU : le benchmark
INT8 produit actuellement `0 partie/s`, alors que FP32 atteint environ
`2,2 parties/s`.

Ne pas utiliser `infer_precision = "int8"` en production tant que ce point
n’est pas réglé.

## À faire plus tard

- [ ] Ajouter/installer le support TensorRT dans le chemin Rust `ort`.
- [ ] Vérifier que le provider TensorRT est réellement chargé et utilisé.
- [ ] Exporter le modèle FP32 sans quantification dynamique `ConvInteger`.
- [ ] Préparer un jeu de calibration représentatif : positions Connect4
      issues du self-play, avec plusieurs phases de partie.
- [ ] Faire une quantification statique calibrée au format QDQ ou via TensorRT.
- [ ] Laisser TensorRT choisir les kernels INT8 pour les convolutions et les
      couches linéaires.
- [ ] Mettre en cache le moteur TensorRT pour éviter sa reconstruction à chaque
      lancement.
- [ ] Vérifier qu’aucun sous-graphe INT8 ne repasse silencieusement sur CPU.
- [ ] Comparer les sorties FP32/INT8 sur les mêmes plateaux : policy, value,
      erreur max et différence de coup choisi.
- [ ] Comparer le niveau de jeu : taux de victoire INT8 contre FP32.
- [ ] Comparer les performances du pipeline complet avec les mêmes paramètres :
      durée, seed, simulations, batch GPU et nombre de workers.
- [ ] Mesurer `parties/s`, `samples/s` et le temps d’inférence pur.
- [ ] Ne remettre `infer_precision = "int8"` par défaut qu’après validation.

## Commandes et benchmarks existants

Benchmark de la GEMM INT8 via le chemin Rust/ORT CUDA :

```powershell
uv run python .\utils\speedtest_onnx.py
```

Benchmark du modèle Connect4 complet via self-play GPU :

```powershell
uv run python .\utils\speedtest_connect4_onnx.py
```

Le second benchmark est le plus important pour valider un vrai gain de
production.

## Points techniques à garder en tête

- Le benchmark `torch._int_mm` mesure principalement une grosse GEMM et ne
  garantit pas un gain équivalent sur les petites convolutions du réseau.
- Le modèle actuel contient principalement des convolutions, pas seulement des
  multiplications matricielles.
- La GTX 1650 dispose du chemin DP4A mais pas de Tensor Cores ; le gain
  TensorRT INT8 devra donc être mesuré, pas supposé.
- La quantification dynamique ajoute des opérations de quantification des
  activations à chaque inférence et peut annuler le bénéfice sur un petit CNN.
