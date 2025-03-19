# Poincare embedding

## Training

```sh
data="wordnet_mammal"
model="poincare"
criterion="poincare_negative-sampling"
optimizer="rsgd"

python local/train.py \
data="${data}" \
model="${model}" \
criterion="${criterion}" \
optimizer="${optimizer}"
```

## Visualization

```sh
data="wordnet_mammal"
model="poincare"

exp_dir="exp/<DATE>"

root="mammal.n.01"

python local/visualize.py \
data="${data}" \
model="${model}" \
exp_dir="${exp_dir}" \
root="${root}"
```
