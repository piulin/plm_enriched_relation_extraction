#!/usr/bin/env zsh

experiment_folder=/mnt/c/Users/gop7rng/Documents/thesis/src/plm_enriched_attention/mlruns/4/

rsync -av --progress gop7rng@si-dl01-login01.de.bosch.com:thesis/src/plm_enriched_attention/mlruns/\
 /mnt/c/Users/gop7rng/Documents/thesis/src/plm_enriched_attention/mlruns/

folders=`find $experiment_folder -mindepth 1 -maxdepth 1 -type d`

folders_after=(`echo $folders | sort | tr '\n' ' '`)

echo "Folders found: $folders_after"

for folder in $folders_after
do
  echo "Updating yaml: $folder"
  sed  -i 's/\/home\/gop7rng\//C:\/Users\/gop7rng\/Documents\//g' $folder/meta.yaml

done
