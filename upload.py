from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_5M",
    repo_id="PY007/EasyContext-64K-h2o",
    repo_type="model"
)