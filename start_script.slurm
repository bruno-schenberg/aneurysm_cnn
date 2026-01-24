#!/bin/bash
#SBATCH --job-name=teste_aneurysm_cnn
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # CPUs para o DataLoader do PyTorch
#SBATCH --gpus-per-task=1      # Solicita 1 GPU para a tarefa
#SBATCH --time=01:00:00

echo -e "\n## Job iniciado em $(date +'%d-%m-%Y as %T') #####################\n"

## Diretório de trabalho temporário (scratch) para melhor performance de I/O
SCRATCH=/scratch/local
WRKDIR=$SCRATCH/$SLURM_JOB_ID

## Informacoes do job impressos no arquivo de saida.
echo -e "## Job ID:                        $SLURM_JOB_ID"
echo -e "## Node de execucao do job:       $(hostname -s)"
echo -e "## Diretorio de submissao do job: $SLURM_SUBMIT_DIR"
echo -e "## Diretorio de scratch do job:   $WRKDIR \n"

## Cria o diretório de scratch para o job.
mkdir -p $WRKDIR

## Copia os arquivos necessários para o diretório de trabalho
## Supondo que 'train_models.py' está em 'training_engine'
cp -r $HOME/aneurysm_cnn/training_engine $WRKDIR/

## Entra no diretório de trabalho
cd $WRKDIR/training_engine

#########################################
##-------  Inicio do trabalho  ------##
#########################################

## Ativa o ambiente Conda
source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn

## Verificação do ambiente e da GPU (Opcional - para log)
echo -e "\n## Verificando ambiente e GPU... ##"
echo "Ambiente Conda ativado: $CONDA_DEFAULT_ENV"
echo "Caminho do Python: $(which python)"
python -c "import torch; print(f'PyTorch - CUDA disponível: {torch.cuda.is_available()}'); print(f'PyTorch - Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
echo -e "## ...Verificação concluída. ##\n"

## Executa o script de treinamento
## O flag -u garante que a saída do Python não seja bufferizada
srun python -u train_models.py

## Limpa o diretório de scratch
rm -rf $WRKDIR

echo -e "\n## Job finalizado em $(date +'%d-%m-%Y as %T') ###################"