# Euclid embedding

```sh
data="wordnet_mammal"
model="euclid"
criterion="euclid_negative-sampling"
optimizer="sgd"

python local/train.py \
data="${data}" \
model="${model}" \
criterion="${criterion}" \
optimizer="${optimizer}"
```
