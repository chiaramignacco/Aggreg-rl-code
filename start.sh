#!/bin/bash
#SBATCH --job-name=projet
#SBATCH --chdir=/home/chiara.mignacco/AGGREG-RL-CODE/ 
# -- Optionnel, pour être notifié par email :
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chiara.mignacco@universite-paris-saclay.fr
# -- Sortie standard et d'erreur dans le fichier .output :
#SBATCH --output=./%j.stdout
#SBATCH --error=./%j.stderr
# -- Contexte matériel
#SBATCH --nodes=1
#SBATCH --nodelist=node22
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=2G
#SBATCH --cpus-per-task=2

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate aggreg
export OMP_NUM_THREADS=2
echo $OMP_NUM_THREADS

## doc
# squeue permet de voir les jobs en attente ou en train de tourner. S'ils tournent, il y aura un R dans la colonne ST.
# sattach permet d'attacher sur le terminal les E/S d'un job en train de tourner. Ça permet de surveiller l'avancée d'un job, ou par exemple d'interagir avec un debugger. ctrl-c permet de détacher de nouveau le job et de le laisser de nouveau tourner en fond (de manière non bloquante).
# scancel permet permet de supprimer une soumission ou d’arrêter le job s'il est en cours d’exécution.
# sstat donne des infos sur les ressources utilisées par un job

srun jupyter nbconvert --execute --to notebook --inplace Aggreg_RL actor critic/matching_problem-expw_NN-Copy2.ipynb > log 2>&1
