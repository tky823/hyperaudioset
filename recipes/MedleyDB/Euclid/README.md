# Euclid embedding

## Training

```sh
data="medleydb"
model="euclid"
criterion="euclid_negative-sampling"
optimizer="sgd"

python local/train.py \
data="${data}" \
model="${model}" \
criterion="${criterion}" \
optimizer="${optimizer}"
```

## Visualization

```sh
data="medleydb"
model="euclid"

exp_dir="exp/<DATE>"

root="musical instrument"

python local/visualize.py \
data="${data}" \
model="${model}" \
exp_dir="${exp_dir}" \
root="${root}"
```
