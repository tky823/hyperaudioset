# Poincare embedding

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
