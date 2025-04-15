export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \

python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --actor \
    --checkpoint_path=first_run \


# python ../../train_rlpd.py "$@" \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=first_run \
#     --actor \