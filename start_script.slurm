#!/bin/bash
#SBATCH --job-name=teste_aneurysm_cnn
#SBATCH --partition=gpu-nvidia
#SBATCH --output=%x_%j.out     # <--- ONDE O TEXTO É SALVO
#SBATCH --error=%x_%j.err      # <--- ONDE OS ERROS SÃO SALVOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # Importante para o DataLoader do PyTorch não gargalar
#SBATCH --time=01:00:00

source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn

cd $HOME/aneurysm_cnn/training_engine


# 3. Verificação (Opcional - para log)
echo "Ambiente ativado: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
# Verifica se o PyTorch vê a GPU
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0)}')"

# O flag -u garante que o output vá para o arquivo .out em tempo real
srun python -u train_models.py