export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../record_success_fail.py "$@" \
    --exp_name=pick_cube_sim \
