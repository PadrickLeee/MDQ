export PYTHONPATH=$PYTHONPATH:/home/ma-user/modelarts/work/jjw/SFT/lqf/1d-tokenizer

# # finetune stage 1 config
# WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_titok.py config=configs/training/TiTok/stage1/finetune_titok_l32.yaml \
#     experiment.project="titok_l32_stage1" \
#     experiment.name="titok_l32_stage1_run2" \
#     experiment.output_dir="titok_l32_stage1_run2" \
#     training.per_gpu_batch_size=64

# finetune stage 2 config
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=8888 --same_network scripts/train_titok.py config=configs/training/TiTok/stage2/finetune_titok_l32.yaml \
    experiment.project="titok_l32_stage2" \
    experiment.name="titok_l32_stage2_run2" \
    experiment.output_dir="titok_l32_stage2_run2" \
    training.per_gpu_batch_size=32