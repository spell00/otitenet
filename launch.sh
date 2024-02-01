#!/usr/bin/env python3

n_epochs=1000  # The number of epochs to train for.
task=Otitis
i=0
for is_stn in 0 1
do
	for is_transform in 0 1
	do
		for dloss in no inverseTriplet
		do
      	cuda=$((i%2)) # Divide by the number of gpus available
		/usr/bin/python3 otitenet/train.py --is_stn=$is_stn --dloss=$dloss --device=cuda:$cuda --classif_loss=ce --n_epochs=$n_epochs --task=$task --is_transform=$is_transform --weighted_sampler=1 --groupkfold=1 &
		i=$((i+1))
    done
	done
done
